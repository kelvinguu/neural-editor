import argparse

from gtd.utils import Config
from textmorph.language_model.training_run import LMTrainingRuns


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('config_path')
args = arg_parser.parse_args()

runs = LMTrainingRuns()
config = Config.from_file(args.config_path)
run = runs.new(config)

run.train()