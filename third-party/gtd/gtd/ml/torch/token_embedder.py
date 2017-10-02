import torch
from gtd.ml.torch.utils import GPUVariable
from torch.nn import Embedding, Module

from gtd.ml.torch.seq_batch import SequenceBatch


class TokenEmbedder(Module):
    """
    Attributes:
        vocab (WordVocab)
        embeds (Variable): of shape (vocab_size, embed_dim)
        embed_dim (int)
    """

    def __init__(self, word_embeddings, trainable=True):
        """Create TokenEmbedder.

        Args:
            word_embeddings (WordEmbeddings)
            trainable (bool): if False, the embedding array will not see
                gradient steps
        """
        super(TokenEmbedder, self).__init__()

        arr = word_embeddings.array  # np.ndarray
        vocab_size, embed_dim = arr.shape

        assert vocab_size == len(word_embeddings.vocab)
        self.vocab = word_embeddings.vocab
        self.embed_dim = embed_dim

        # create Embedding Module
        vocab_size, embed_dim = arr.shape
        self._embedding = TrainFlagEmbedding(
            vocab_size, embed_dim, arr, trainable=trainable)

    @property
    def embeds(self):
        """Return Variable of shape (vocab_size, embed_dim)."""
        return self._embedding.weight

    def embed_indices(self, indices):
        """Embed array of indices.

        Args:
            indices (Variable[LongTensor]): of shape (X1, X2) or (X1)

        Returns:
            embeds (Variable[FloatTensor]): of shape (X1, X2, embed_dim) or (X1, embed_dim)
        """
        return self._embedding(indices)

    def embed_seq_batch(self, seq_batch):
        """Embed elements of a SequenceBatch.

        Args:
            seq_batch (SequenceBatch)

        Returns:
            SequenceBatch
        """
        return SequenceBatch(self._embedding(seq_batch.values), seq_batch.mask)

    def embed_tokens(self, tokens):
        """Embed list of tokens.

        Args:
            tokens (list[unicode])

        Returns:
            embeds (Variable[FloatTensor]): of shape (len(tokens), embed_dim)
        """
        vocab = self.vocab
        indices = GPUVariable(torch.LongTensor([vocab.word2index(t) for t in tokens]))
        return self._embedding(indices)


class TrainFlagEmbedding(Module):
    """Small wrapper around PyTorch Embedding object. Exports a trainable
    flag, which allows you to fix the weights matrix. Obeys same interface as
    PyTorch Embedding object, except for extra constructor arguments.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 initial_embeddings, **kwargs):
        """Constructs TrainFlagEmbedding with embeddings initialized with
        initial_embeddings.

        Args:
            num_embeddings (int)
            embedding_dim (int)
            initial_embeddings (np.array): (num_embeddings, embedding_dim)
            trainable (bool): if False, weights matrix will not change.
                (default True)
            kwargs: all other supported keywords in torch.nn.Embeddings.
        """
        super(TrainFlagEmbedding, self).__init__()
        trainable = kwargs.pop("trainable", True)
        self._trainable = trainable
        if trainable:
            embedding = Embedding(
                num_embeddings, embedding_dim, **kwargs)
            embedding.weight.data.set_(
                torch.from_numpy(initial_embeddings))
            self._embedding = embedding
            self._weight = embedding.weight
        else:
            self._weight = GPUVariable(
                torch.from_numpy(initial_embeddings))

    @property
    def weight(self):
        return self._weight

    def forward(self, index):
        """Looks up a batch of indices.

        Args:
            index (Variable[LongTensor]): (batch, indices per batch)

        Returns:
            Tensor: (batch, indices per batch, embedding_dim)
        """
        if self._trainable:
            return self._embedding(index)
        else:
            batch, num_indices = index.size()
            flattened_index = index.view(batch * num_indices)
            embeddings = torch.index_select(
                self._weight, 0, flattened_index)
            return embeddings.view(batch, num_indices, -1)
