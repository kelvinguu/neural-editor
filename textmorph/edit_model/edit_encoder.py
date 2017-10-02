import torch
from torch.nn import Module, Linear, Parameter, Hardtanh
from gtd.ml.torch.utils import GPUVariable
from gtd.ml.torch.seq_batch import SequenceBatch
import numpy as np

class EditEncoder(Module):
    """
    EditEncoder maps insert / delete embeddings into a single edit vector of dimensionality edit_dim
    """
    def __init__(self, word_dim, edit_dim, kappa_init, norm_eps, norm_max):
        super(EditEncoder, self).__init__()
        self.linear = Linear(edit_dim, edit_dim)
        self.linear_prenoise = Linear(word_dim, edit_dim/2, bias=False)
        self.noise_scaler = kappa_init
        self.norm_eps = norm_eps
        self.norm_max = norm_max
        self.normclip = Hardtanh(0, self.norm_max - norm_eps)

    def forward(self, insert_embeds, insert_embeds_exact, delete_embeds, delete_embeds_exact, draw_samples = False, draw_p = False):
        """Create agenda vector.

        Args:
            insert_embeds (SequenceBatch): of shape (batch_size, max_edits, word_dim)
            insert_embeds_exact (SequenceBatch): of shape (batch_size, max_edits, word_dim)
            delete_embeds (SequenceBatch): of shape (batch_size, max_edits, word_dim)
            delete_embeds_exact (SequenceBatch): of shape (batch_size, max_edits, word_dim)
            draw_samples (bool) : flag for whether to add noise for variational approx. disable at test time.

        Returns:
            edit_embed (Variable): of shape (batch_size, edit_vec_cim)
        """
        insert_embed = SequenceBatch.reduce_sum(insert_embeds)  # (batch_size, word_dim)
        insert_embed += SequenceBatch.reduce_sum(insert_embeds_exact)  # (batch_size, word_dim)
        delete_embed = SequenceBatch.reduce_sum(delete_embeds)  # (batch_size, word_dim)
        delete_embed += SequenceBatch.reduce_sum(delete_embeds_exact)  # (batch_size, word_dim)
        insert_set = self.linear_prenoise(insert_embed)
        delete_set = self.linear_prenoise(delete_embed)
        combined_map = torch.cat([insert_set, delete_set], 1)
        if draw_samples:
            if draw_p:
                batch_size, edit_dim = combined_map.size()
                combined_map = self.draw_p_noise(batch_size, edit_dim)
            else:
                combined_map = self.sample_vMF(combined_map, self.noise_scaler)
        edit_embed = combined_map
        return edit_embed

    def seq_batch_noise(self, seq_batch, draw_noise):
        """
        Returns a noisy version of seq_batch, in which every vector is noisy and unit norm.
        :param seq_batch(SequenceBatch): a sequence batch of elements
        :return: noisy version of seq-batch
        """
        values = seq_batch.values
        mask = seq_batch.mask
        batch_size, max_edits, w_embed_size = values.size()
        new_values = GPUVariable(torch.from_numpy(np.zeros((batch_size, max_edits, w_embed_size),dtype=np.float32)))
        phint = self.sample_vMF(values[:,0,:], self.noise_scaler)
        prand = self.draw_p_noise(batch_size, w_embed_size)
        m_expand = mask.expand(batch_size, w_embed_size)
        new_values[:, 0, :] = phint*m_expand+ prand*(1-m_expand)
        return SequenceBatch(values=new_values*draw_noise, mask=mask)

    def draw_p_noise(self, batch_size, edit_dim):
        rand_draw = GPUVariable(torch.randn(batch_size, edit_dim))
        rand_draw = rand_draw / torch.norm(rand_draw, p=2, dim=1).expand(batch_size, edit_dim)
        rand_norms = (torch.rand(batch_size,1)*self.norm_max).expand(batch_size, edit_dim)
        return rand_draw * GPUVariable(rand_norms)


    def add_norm_noise(self, munorm, eps):
        """
        KL loss is - log(maxvalue/eps)
        cut at maxvalue-eps, and add [0,eps] noise.
        """
        trand = torch.rand(1).expand(munorm.size())*eps
        return (self.normclip(munorm) + GPUVariable(trand))

    def sample_vMF(self, mu, kappa):
        """vMF sampler in pytorch.

        http://stats.stackexchange.com/questions/156729/sampling-from-von-mises-fisher-distribution-in-python

        Args:
            mu (Tensor): of shape (batch_size, 2*word_dim)
            kappa (Float): controls dispersion. kappa of zero is no dispersion.
        """
        batch_size, id_dim = mu.size()
        result_list = []
        for i in range(batch_size):
            munorm = mu[i].norm().expand(id_dim)
            munoise = self.add_norm_noise(munorm, self.norm_eps)
            if float(mu[i].norm().data.cpu().numpy()) > 1e-10:
                # sample offset from center (on sphere) with spread kappa
                w = self._sample_weight(kappa, id_dim)
                wtorch = GPUVariable(w*torch.ones(id_dim))

                # sample a point v on the unit sphere that's orthogonal to mu
                v = self._sample_orthonormal_to(mu[i]/munorm, id_dim)

                # compute new point
                scale_factr = torch.sqrt(GPUVariable(torch.ones(id_dim)) - torch.pow(wtorch,2))
                orth_term = v * scale_factr
                muscale = mu[i] * wtorch / munorm
                sampled_vec = (orth_term + muscale)*munoise
            else:
                rand_draw = GPUVariable(torch.randn(id_dim))
                rand_draw = rand_draw / torch.norm(rand_draw, p=2).expand(id_dim)
                rand_norms = (torch.rand(1) * self.norm_eps).expand(id_dim)
                sampled_vec = rand_draw*GPUVariable(rand_norms)#mu[i]
            result_list.append(sampled_vec)

        return torch.stack(result_list,0)

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa) # b= 1/(sqrt(4.* kdiv**2 + 1) + 2 * kdiv)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x ** 2)  # dim * (kdiv *x + np.log(1-x**2))

        while True:
            z = np.random.beta(dim / 2., dim / 2.)  #concentrates towards 0.5 as d-> inf
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u): #thresh is dim *(kdiv * (w-x) + log(1-x*w) -log(1-x**2))
                return w

    def _sample_orthonormal_to(self, mu, dim):
        """Sample point on sphere orthogonal to mu.
        """
        v = GPUVariable(torch.randn(dim))
        rescale_value = mu.dot(v) / mu.norm()
        proj_mu_v = mu * rescale_value.expand(dim)
        ortho = v - proj_mu_v
        ortho_norm = torch.norm(ortho)
        return ortho / ortho_norm.expand_as(ortho)

def test_sample_weight(kappa, dim):
    """Rejection sampling scheme for sampling distance from center on
    surface of the sphere.
    """
    dim = dim - 1  # since S^{n-1}
    b = dim / (np.sqrt(4. * kappa ** 2 + dim ** 2) + 2 * kappa)
    x = (1. - b) / (1. + b)
    c = kappa * x + dim * np.log(1 - x ** 2)

    while True:
        z = np.random.beta(dim / 2., dim / 2.)
        w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
        u = np.random.uniform(low=0, high=1)
        if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
            return w

def get_ev(kappa,dim,nsamp):
    samp_in = np.array([test_sample_weight(kappa,dim) for i in xrange(nsamp)])
    return np.mean(samp_in), np.std(samp_in), np.percentile(samp_in, np.arange(0,100,10))

def get_mode(kappa,dim):
    return np.sqrt(4*(kappa**2.0)+dim**2.0+6*dim+9)/(2*kappa) - (dim+3.0)/(2*kappa)