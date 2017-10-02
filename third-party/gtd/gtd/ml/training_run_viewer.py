import json
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import pwd
from IPython.core.display import display, HTML
from os.path import join, basename

import os
from os import listdir
from prettytable import PrettyTable

from gtd.chrono import verboserate
from gtd.log import in_ipython, jupyter_no_margins, Metadata


class TrainingRunViewer(object):
    def __init__(self, runs):
        """Construct TrainingRunViewer.
        
        Args:
            runs (gtd.ml.TrainingRuns)
        """
        self._runs = runs
        self._renderers = OrderedDict()

    def add(self, name, renderer, post_processor=None):
        """Add a renderer.
        
        Args:
            name (unicode): name for the attribute
            renderer (Callable[str, object]): takes a run dir (absolute path) and returns something to print.
            post_processor (Callable[object, unicode]): takes the output of the renderer and returns a modified output.

        Returns:

        """
        if post_processor:
            r = lambda path: post_processor(renderer(path))
        else:
            r = renderer
        self._renderers[name] = r

    def view(self, select=lambda path: True):
        """View runs.
        
        Args:
            select (Callable[str, bool]): given a path to a run, returns True if we want to display the
                run, False otherwise.
        """
        field_names = self._renderers.keys()
        table = PrettyTable(field_names=field_names)
        types = OrderedDict((n, set()) for n in field_names)

        for i, path in verboserate(self._runs._int_dirs.items(), desc='Scanning runs.'):
            if not select(path):
                continue

            row = []
            for render in self._renderers.values():
                try:
                    s = render(path)
                except:
                    s = u''
                row.append(s)

            # record types
            for name, elem in zip(field_names, row):
                types[name].add(type(elem))

            table.add_row(row)

        self._print_table(table)

        # display types for each attribute
        type_table = PrettyTable(['attribute', 'types'])
        for name, type_set in types.iteritems():
            type_table.add_row([name, ', '.join(t.__name__ for t in type_set)])
        self._print_table(type_table)

    @classmethod
    def _print_table(cls, table):
        if in_ipython():
            jupyter_no_margins()
            display(HTML(table.get_html_string()))
        else:
            print table


class Renderer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __call__(self, path):
        """Render.

        Args:
            path (str): absolute path to a run directory

        Returns:
            object: value to be displayed in a pretty-printed table. Should implement __str__ and __unicode__.
        """
        raise NotImplementedError


# Some renderers below are just functions, for simplicity.


class JSONSelector(Renderer):
    def __init__(self, file_path, json_keys):
        """Select a value in a JSON file, or a HOCON file.

        Args:
            file_path (str): path to the JSON file, relative to run dir root.
            json_keys (list[str]): path from the root of the JSON tree to the target attribute
        """
        self.file_path = file_path
        self.json_keys = json_keys

    def __call__(self, path):
        full_path = join(path, self.file_path)
        try:
            # try loading as JSON
            with open(full_path, 'r') as f:
                x = json.load(f)
        except ValueError:
            # try loading as HOCON
            x = Metadata.from_file(full_path, fmt='hocon')

        for key in self.json_keys:
            x = x[key]

        return x


class Commit(Renderer):
    def __init__(self):
        self._commit = JSONSelector('metadata.txt', ['commit'])
        self._dirty = JSONSelector('metadata.txt', ['dirty_repo'])

    def __call__(self, path):
        c = self._commit(path)[:8]
        d = ' (dirty)' if self._dirty(path) else ''
        return '{}{}'.format(c, d)


class NumSteps(Renderer):
    def __init__(self):
        self.json_selector = JSONSelector('metadata.txt', ['steps'])

    def __call__(self, path):
        try:
            steps = self.json_selector(path)  # try looking in JSON
        except:
            # if that fails, look at the largest checkpoint
            ckpt_nums = checkpoint_numbers(join(path, 'checkpoints'))
            steps = max(ckpt_nums) if ckpt_nums else 0
        return steps


class Owner(Renderer):
    def __init__(self, user_ids):
        self.user_ids = user_ids

    def __call__(self, path):
        stat_info = os.stat(path)
        uid = stat_info.st_uid
        try:
            user = pwd.getpwuid(uid)[0]
        except:
            # sometimes no name is associated with the ID
            user = self.user_ids.get(uid, uid)

        return str(user)


def checkpoint_numbers(checkpoints_dir):
    """Return the train steps at which checkpoints were saved (sorted ascending)."""
    dirs = [d for d in listdir(checkpoints_dir) if d.endswith('.checkpoint')]
    return sorted([int(d[:-11]) for d in dirs])


def run_name(path):
    return basename(path)


def num_checkpoints(path):
    ckpt_nums = checkpoint_numbers(join(path, 'checkpoints'))
    return len(ckpt_nums)
