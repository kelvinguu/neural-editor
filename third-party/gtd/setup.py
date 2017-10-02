#!/usr/bin/env python
from distutils.core import setup, Command


class Test(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['py.test', '-v', '--doctest-modules', 'gtd'])
        raise SystemExit(errno)


setup(name='gtd',
      version='1.0',
      packages=['gtd'],
      description='Get things done.',
      cmdclass={'test': Test},
)