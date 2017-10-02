import io
import logging
import socket
from abc import ABCMeta, abstractmethod
from collections import Mapping
from os.path import join

from git import Repo
from tensorboard_logger import tensorboard_logger

from gtd.io import IntegerDirectories, Workspace
from gtd.log import SyncedMetadata
from gtd.utils import Config, cached_property


class TrainingRunWorkspace(Workspace):
    def __init__(self, root):
        super(TrainingRunWorkspace, self).__init__(root)
        for attr in ['config', 'metadata']:
            self.add_file(attr, '{}.txt'.format(attr))
        for attr in ['git_patches', 'tensorboard']:
            self.add_dir(attr, attr)


class TrainingRun(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, save_dir):
        """Create TrainingRun.

        Args:
            config (Config)
            save_dir (str)
        """
        self._config = config
        self._workspace = TrainingRunWorkspace(save_dir)
        self.metadata['host'] = socket.gethostname()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @property
    def config(self):
        return self._config

    @property
    def workspace(self):
        return self._workspace

    @cached_property
    def metadata(self):
        return SyncedMetadata(self.workspace.metadata, fmt='json')

    @cached_property
    def tb_logger(self):
        return tensorboard_logger.Logger(self.workspace.tensorboard)

    def record_commit(self, src_dir):
        repo = Repo(src_dir)

        if 'dirty_repo' in self.metadata or 'commit' in self.metadata:
            raise RuntimeError('A commit has already been recorded.')

        self.metadata['dirty_repo'] = repo.is_dirty()
        self.metadata['commit'] = repo.head.object.hexsha.encode('utf-8')

    def dump_diff(self, src_dir):
        repo = Repo(src_dir)
        diffindex = repo.head.commit.diff(None, create_patch=True)
        if len(diffindex) > 0:
            print 'uncomitted changes being stored as patches'
            patch_strings = [unicode(diff) for diff in diffindex]
            patch_filenames = [unicode(diff.a_rawpath).replace(u'/', u'-').replace(u'.', u'-') + u'.patch' for diff in
                               diffindex]
            for strin, filename in zip(patch_strings, patch_filenames):
                file_out = join(self.workspace.git_patches, filename)
                with io.open(file_out, 'w', encoding='utf-8') as fout:
                    fout.writelines(strin)
        else:
            print 'no changes to diff. ignoring git diff.'

    def match_commit(self, src_dir):
        """Check that the current commit matches the recorded commit for this run.

        Raises an error if commits don't match, or if there is dirty state.

        Args:
            src_dir (str): path to the Git repository
        """
        if self.metadata['dirty_repo']:
            raise EnvironmentError('Working directory was dirty when commit was recorded.')

        repo = Repo(src_dir)
        if repo.is_dirty():
            raise EnvironmentError('Current working directory is dirty.')

        current_commit = repo.head.object.hexsha.encode('utf-8')
        run_commit = self.metadata['commit']
        if current_commit != run_commit:
            raise EnvironmentError("Commits don't match.\nCurrent: {}\nRecorded: {}".format(current_commit, run_commit))


class TrainingRuns(Mapping):
    """A map from integers to TrainingRuns."""

    def __init__(self, root_dir, src_dir, run_factory, check_commit=True):
        """Create TrainingRuns object.

        Args:
            root_dir (str): directory where all training run data will be stored
            src_dir (str): a Git repository path (used to check commits)
            run_factory (Callable[[Config, str], TrainingRun]): a Callable, which takes a Config and a save_dir
                as arguments, and creates a new TrainingRun.
            check_commit (bool): if True, checks that current working directory is on same commit as when the run
                was originally created.
        """
        self._int_dirs = IntegerDirectories(root_dir)
        self._src_dir = src_dir
        self._run_factory = run_factory
        self._check_commit = check_commit

    def _config_path(self, save_dir):
        return join(save_dir, 'config.txt')

    def __getitem__(self, i):
        """Reload an existing TrainingRun."""
        save_dir = self._int_dirs[i]
        config = Config.from_file(self._config_path(save_dir))
        run = self._run_factory(config, save_dir)
        if self._check_commit:
            run.match_commit(self._src_dir)

        logging.info('Reloaded TrainingRun #{}'.format(i))
        return run

    def new(self, config, name=None):
        """Create a new TrainingRun."""
        print 'TrainingRun configuration:\n{}'.format(config)

        save_dir = self._int_dirs.new_dir(name=name)
        cfg_path = self._config_path(save_dir)
        config.to_file(cfg_path)  # save the config
        run = self._run_factory(config, save_dir)
        run.record_commit(self._src_dir)
        run.dump_diff(self._src_dir)
        run.metadata['config'] = config._config_tree  # save config in metadata, for programmatic access

        print 'New TrainingRun created at: {}'.format(run.workspace.root)
        return run

    def __iter__(self):
        return iter(self._int_dirs)

    def __len__(self):
        return len(self._int_dirs)

    def paths(self):
        return self._int_dirs.values()
