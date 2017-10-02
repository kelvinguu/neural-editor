from gtd.ml.training_run import TrainingRun
from gtd.utils import cached_property


class TFTrainingRun(TrainingRun):
    def __init__(self, config, save_dir):
        super(TFTrainingRun, self).__init__(config, save_dir)

    @cached_property
    def saver(self):
        from gtd.ml.tf.utils import Saver
        return Saver(self.workspace.checkpoints, keep_checkpoint_every_n_hours=5)