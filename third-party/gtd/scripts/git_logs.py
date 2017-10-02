from git import Repo
from os.path import join

import sys
print sys.path

from gtd.git_utils import commit_diff
from gtd.chrono import verboserate


repo_path = sys.argv[1]
max_count = sys.argv[2]
files = set(sys.argv[3:])

def format_commit(c):
    msg = c.message.split('\n')[0]
    return '{}\t{}'.format(c.hexsha, msg)

repo = Repo(repo_path)
commits = list(repo.iter_commits('master', max_count=max_count))
lines = []
for c in verboserate(commits, desc='Scanning commits', total=max_count):
    if len(files & commit_diff(c)) == 0:
        continue
    lines.append(format_commit(c))

log_path = join(repo_path, 'git-logs.tsv')
with open(log_path, 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')