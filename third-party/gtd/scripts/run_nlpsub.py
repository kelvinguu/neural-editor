#!/u/nlp/packages/anaconda2/bin/python

# THIS SCRIPT SHOULD BE SYMLINKED INTO THE ROOT OF YOUR GIT REPO
# It assumes that config.json and run_docker.py can also be found at the root of your repo.

import argparse
import json
import os
from datetime import datetime
import subprocess

from os.path import abspath, dirname, join

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-n', '--name', default='unnamed', help='Job name.')
arg_parser.add_argument('-t', '--tail', action='store_true', help='Tail the output.')
arg_parser.add_argument('-x', '--debug', action='store_true', help='Print command instead of running.')
arg_parser.add_argument('-m', '--host', action='append', help='Allowed hosts.')
arg_parser.add_argument('-g', '--gpu', default='0', help='GPU to use.')
arg_parser.add_argument('command', nargs='+', help='Command passed to run_docker.py')
args = arg_parser.parse_args()

repo_dir = abspath(dirname(__file__))
with open(join(repo_dir, 'config.json'), 'r') as f:
    config = json.load(f)
data_env_var = config['data_env_var']  # environment variable used by code to locate data, e.g. 'TEXTMORPH_DATA'
data_dir = os.environ[data_env_var]

time_str = datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
log_dir = join(data_dir, 'nlpsub', '{}_{}'.format(args.name, time_str))

nlpsub_options = ['--queue=jag',
                  '--cores=1',
                  '--mem=2g',
                  '--priority=high',
                  '--name={}'.format(args.name),
                  '--log-dir={}'.format(log_dir),
                  '--mail=bea',
                  '--clobber',
                  '--verbose']
if args.tail:
    nlpsub_options.append('--tail')

if args.host is not None:
    nlpsub_options.append('--hosts={}'.format(','.join(args.host)))

def bash_string(s):
    s = s.replace('\\', '\\\\')  # \ -> \\
    s = s.replace('\"', '\\\"')  # " -> \"
    return '\"{}\"'.format(s)  # s -> "s"


cmd = ' '.join(args.command)

docker_cmd = '/u/nlp/packages/anaconda2/bin/python run_docker.py -r -g {gpu} {command}'.format(gpu=args.gpu, command=bash_string(cmd))

nlpsub_cmd = 'nlpsub {options} {command}'.format(options=' '.join(nlpsub_options), command=bash_string(docker_cmd))

print 'Logging to: {}'.format(log_dir)
print 'Allowed hosts: {}'.format(args.host)
print 'GPU to use: {}'.format(args.gpu)
print 'Command inside Docker: {}'.format(cmd)
print nlpsub_cmd
print

if not args.debug:
    subprocess.call(nlpsub_cmd, shell=True)
