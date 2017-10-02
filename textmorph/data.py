import os
from gtd.io import Workspace


# Set location of local data directory from environment variable
env_var = 'TEXTMORPH_DATA'
if env_var not in os.environ:
    assert False, env_var + ' environmental variable must be set.'
root = os.environ[env_var]

# define workspace
workspace = Workspace(root)

# config
workspace.add_file('config', 'config.txt')

# Training runs
workspace.add_dir('edit_runs', 'edit_runs')
workspace.add_dir('lm_runs', 'lm_runs')
workspace.add_dir('retriever_runs', 'retriever_runs')

# user IDs
workspace.add_file('user_ids', 'user_ids.json')

# word vectors
workspace.add_dir('word_vectors', 'word_vectors')

# nearest neighbors
workspace.add_dir('nearest_sentences', 'nearest_sentences')