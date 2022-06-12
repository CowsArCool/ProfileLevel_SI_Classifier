import yaml
import wandb
import os.path as path
from train_reddit import wandb_sweep as reddit_sweep
from train_twitter import wandb_sweep as twitter_sweep
# from sweeptest import main
import pprint


COUNT = 30
PROJECT = 'twitter'

RUNNING_DIR = path.dirname(path.realpath(__file__))
with open(path.join(RUNNING_DIR, '{}_sweep.yaml'.format(PROJECT))) as f:
    sweep_config = yaml.safe_load(f)

pprint.pprint(sweep_config)


if PROJECT == 'reddit':
    sweep_id = wandb.sweep(
        sweep_config, project="BERT Implicitly Labeled Reddit v2")
    wandb.agent(sweep_id, reddit_sweep, count=COUNT)

elif PROJECT == 'twitter':
    sweep_id = wandb.sweep(
        sweep_config, project="TwitterSI Classification")
    wandb.agent(sweep_id, twitter_sweep, count=COUNT)

else:
    print('specified project is not available')
