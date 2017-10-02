from datetime import datetime
import numpy as np
from torch.nn.utils import clip_grad_norm

from gtd.ml.training_run import TrainingRun
from gtd.ml.torch.checkpoints import Checkpoints
from gtd.utils import cached_property


class TorchTrainingRun(TrainingRun):
    def __init__(self, config, save_dir):
        super(TorchTrainingRun, self).__init__(config, save_dir)
        self.workspace.add_dir('checkpoints', 'checkpoints')

    @cached_property
    def checkpoints(self):
        return Checkpoints(self.workspace.checkpoints)

    @classmethod
    def _finite_grads(cls, parameters):
        """Check that all parameter gradients are finite.

        Args:
            parameters (List[Parameter])

        Return:
            bool
        """
        for param in parameters:
            if param.grad is None: continue
            if not np.isfinite(param.grad.data.sum()):
                return False
        return True

    @classmethod
    def _take_grad_step(cls, train_state, loss, max_grad_norm=float('inf')):
        """Try to take a gradient step w.r.t. loss.
        
        If the gradient is finite, takes a step. Otherwise, does nothing.
        
        Args:
            train_state (TrainState)
            loss (Variable): a differentiable scalar variable
            max_grad_norm (float): gradient norm is clipped to this value.
        
        Returns:
            bool: True if the gradient was finite.
        """
        model, optimizer = train_state.model, train_state.optimizer
        optimizer.zero_grad()
        loss.backward()

        # clip according to the max allowed grad norm
        grad_norm = clip_grad_norm(model.parameters(), max_grad_norm, norm_type=2)
        # (this returns the gradient norm BEFORE clipping)

        # track the gradient norm over time
        train_state.track_grad_norms(grad_norm)

        finite_grads = cls._finite_grads(model.parameters())

        # take a step if the grads are finite
        if finite_grads:
            optimizer.step()

        # increment step count
        train_state.increment_train_steps()

        return finite_grads

    def _update_metadata(self, train_state):
        self.metadata['last_seen'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.metadata['steps'] = train_state.train_steps
        self.metadata['max_grad_norm'] = train_state.max_grad_norm