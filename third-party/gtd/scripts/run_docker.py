#!/u/nlp/packages/anaconda2/bin/python

# THIS SCRIPT SHOULD BE SYMLINKED INTO THE ROOT OF YOUR GIT REPO
# It assumes that config.json can also be found at the root of your repo.

import argparse
import json
import os

from os.path import dirname, abspath, join
import subprocess

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-r', '--root', action='store_true', help='Run as root in Docker.')
arg_parser.add_argument('-g', '--gpu', default='', help='GPU to use.')
arg_parser.add_argument('-d', '--debug', action='store_true', help='Print command instead of running.')
arg_parser.add_argument('command', nargs='?', default=None,
                        help='Command to execute in Docker. If no command is specified, ' \
                             'you enter interactive mode. ' \
                             'To execute a command with spaces, wrap ' \
                             'the entire command in quotes.')
args = arg_parser.parse_args()

repo_dir = abspath(dirname(__file__))

with open(join(repo_dir, 'config.json'), 'r') as f:
    config = json.load(f)

image = config['docker_image']  # name of the Docker image, e.g. 'kelvinguu/textmorph:1.0'
data_env_var = config['data_env_var']  # environment variable used by code to locate data, e.g. 'TEXTMORPH_DATA'
data_dir = os.environ[data_env_var]

my_uid = subprocess.check_output(['echo', '$UID']).strip()

docker_args = ["--net host",  # access to the Internet
               "--publish 8888:8888",  # only certain ports are exposed
               "--publish 6006:6006",
               "--publish 8080:8080",
               "--ipc=host",
               "--rm",
               "--volume {}:/data".format(data_dir),
               "--volume {}:/code".format(repo_dir),
               "--env {}=/data".format(data_env_var),
               "--env PYTHONPATH=/code",
               "--env NLTK_DATA=/data/nltk",
               "--env CUDA_VISIBLE_DEVICES={}".format(args.gpu),
               "--workdir /code"]

# interactive mode
if args.command is None:
    docker_args.append('--interactive')
    docker_args.append('--tty')
    args.command = '/bin/bash'

if not args.root:
    docker_args.append('--user={}'.format(my_uid))

if args.gpu == '':
    # run on CPU
    docker = 'docker'
else:
    # run on GPU
    docker = 'nvidia-docker'

pull_cmd = "docker pull {}".format(image)

run_cmd = '{docker} run {options} {image} {command}'.format(docker=docker,
                                                            options=' '.join(docker_args),
                                                            image=image,
                                                            command=args.command)
print 'Data directory: {}'.format(data_dir)
print 'Command to run inside Docker: {}'.format(args.command)

print pull_cmd
print run_cmd
if not args.debug:
    subprocess.call(pull_cmd, shell=True)
    subprocess.call(run_cmd, shell=True)
