import json
import logging
import math
import numbers
import os
import platform
import resource
import sys
from collections import MutableMapping
from contextlib import contextmanager
from os.path import join

from IPython.core.display import display, HTML
from pyhocon import ConfigFactory
from pyhocon import ConfigMissingException
from pyhocon import ConfigTree
from pyhocon import HOCONConverter

from gtd.utils import NestedDict, Config


def in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def print_with_fonts(tokens, sizes, colors, background=None):

    def style(text, size=12, color='black'):
        return u'<span style="font-size: {}px; color: {};">{}</span>'.format(size, color, text)

    styled = [style(token, size, color) for token, size, color in zip(tokens, sizes, colors)]
    text = u' '.join(styled)

    if background:
        text = u'<span style="background-color: {};">{}</span>'.format(background, text)

    display(HTML(text))


def gb_used():
    used = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != 'Darwin':
        # on Linux, used is in terms of kilobytes
        power = 2
    else:
        # on Mac, used is in terms of bytes
        power = 3
    return float(used) / math.pow(1024, power)


class Metadata(MutableMapping):
    """A wrapper around ConfigTree.

    Supports a name_scope contextmanager.
    """
    def __init__(self, config_tree=None):
        if config_tree is None:
            config_tree = ConfigTree()

        self._config_tree = config_tree
        self._namestack = []

    @contextmanager
    def name_scope(self, name):
        self._namestack.append(name)
        yield
        self._namestack.pop()

    def _full_key(self, key):
        return '.'.join(self._namestack + [key])

    def __getitem__(self, key):
        try:
            val = self._config_tree.get(self._full_key(key))
        except ConfigMissingException:
            raise KeyError(key)

        if isinstance(val, ConfigTree):
            return Metadata(val)
        return val

    def __setitem__(self, key, value):
        """Put a value (key is a dot-separated name)."""
        self._config_tree.put(self._full_key(key), value)

    def __delitem__(self, key):
        raise NotImplementedError()

    def __iter__(self):
        return iter(self._config_tree)

    def __len__(self):
        return len(self._config_tree)

    def __repr__(self):
        return self.to_str()

    def to_str(self, fmt='hocon'):
        return HOCONConverter.convert(self._config_tree, fmt)

    def to_file(self, path, fmt='hocon'):
        with open(path, 'w') as f:
            f.write(self.to_str(fmt))

    @classmethod
    def from_file(cls, path, fmt='hocon'):
        if fmt == 'hocon':
            config_tree = ConfigFactory.parse_file(path)
        elif fmt == 'json':
            with open(path, 'r') as f:
                d = json.load(f)
            config_tree = ConfigFactory.from_dict(d)
        else:
            raise ValueError('Invalid format: {}'.format(fmt))

        return cls(config_tree)


class SyncedMetadata(Metadata):
    """A Metadata object which writes to file after every change."""
    def __init__(self, path, fmt='hocon'):
        if os.path.exists(path):
            m = Metadata.from_file(path, fmt)
        else:
            m = Metadata()

        super(SyncedMetadata, self).__init__(m._config_tree)
        self._path = path
        self._fmt = fmt

    def __setitem__(self, key, value):
        super(SyncedMetadata, self).__setitem__(key, value)
        self.to_file(self._path, fmt=self._fmt)


def print_list(l):
    for item in l:
        print item


def print_no_newline(s):
    sys.stdout.write(s)
    sys.stdout.flush()


def set_log_level(level):
    """Set the log-level of the root logger of the logging module.

    Args:
        level: can be an integer such as 30 (logging.WARN), or a string such as 'WARN'
    """
    if isinstance(level, str):
        level = logging._levelNames[level]

    logger = logging.getLogger()  # gets root logger
    logger.setLevel(level)


def jupyter_no_margins():
    """Cause Jupyter notebook to take up 100% of window width."""
    display(HTML("<style>.container { width:100% !important; }</style>"))


class TraceSession(object):
    def __init__(self, tracer):
        self.tracer = tracer
        self._values = {}

    @property
    def values(self):
        return self._values

    def save(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.values, f, indent=4, sort_keys=True)

    def __enter__(self):
        if self.tracer._current_session:
            raise RuntimeError('Already in the middle of a TraceSession')

        # register as the current session
        self.tracer._current_session = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # un-register
        self.tracer._current_session = None


class Tracer(object):
    """Log values computed during program execution.
    
    Values are logged to the currently active TraceSession object.
    """
    def __init__(self):
        self._current_session = None

    def session(self):
        return TraceSession(self)

    def log(self, logging_callback):
        """If we are in a TraceSession, execute the logging_callback.
        
        The logging_callback should take a `values` dict as its only argument, and modify `values` in some way.
        
        Args:
            logging_callback (Callable[dict]): a function which takes a `values` dict as its only argument.
        """
        if self._current_session is None:
            return
        logging_callback(self._current_session.values)

    def log_put(self, name, value):
        """Log a value.
        
        Args:
            name (str): name of the variable
            value (object)
        """
        def callback(values):
            if name in values:
                raise RuntimeError('{} already logged'.format(name))
            values[name] = value

        return self.log(callback)

    def log_append(self, name, value):
        """Append a value.

        Args:
            name (str): name of the variable
            value (object): value to append
        """
        def callback(values):
            if name not in values:
                values[name] = []
            values[name].append(value)

        return self.log(callback)


def indent(s, spaces=4):
    whitespace = u' ' * spaces
    return u'\n'.join(whitespace + line for line in s.split(u'\n'))
