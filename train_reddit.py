#
# Default Packages
import os
import sys
import pickle
import numpy as np
import pandas as pd
import os.path as path

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
# Using standard huggingface tokenizer for compatability
from transformers import (BertTokenizer, BertModel,
                          get_linear_schedule_with_warmup)

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1, auroc, recall, precision
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

# Internal Packages
from datasets import RedditImplicitDataModule
from models import *
from utils import *
#


def read_redditSI(data_path, n_examples=None):
    # Data Loading
    reddit_df = pd.read_csv(data_path)[['text', 'class']]

    if n_examples is None:
        n_examples = len(reddit_df)
    reddit_df = reddit_df.sample(n_examples).reset_index(drop=True)
    reddit_df.rename(columns={'class': 'label'}, inplace=True)

    return reddit_df


def main(hparams=None, RUNNING_DIR=os.path.dirname(path.realpath(__file__))):
    datasets_dir = path.join(RUNNING_DIR, 'Datasets')

    reddit_df = read_redditSI(
        path.join(datasets_dir, 'Implicitly_Labeled_Suicide_Reddit.csv'),
        n_examples=hparams['N_EXAMPLES']
    )

    # Tokenization and Batching
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    reddit_data_module = RedditImplicitDataModule(
        reddit_df,
        tokenizer,
        splits=[0.8, 0.2],
        max_example_len=hparams.MAX_EXAMPLE_LEN,
        shuffle=True,
        batch_size=hparams.BATCH_SIZE,
    )

    training_steps = (len(reddit_df)//hparams.BATCH_SIZE)*hparams.NUM_EPOCHS

    model = SuicideClassifier(
        output_classes=hparams.CLASSES,
        training_steps=training_steps,
        warmup_steps=training_steps/5,
        lr=hparams.LEARNING_RATE,
        metrics=['ROC', 'binary_report']
    )

    #
    trainer_params = generate_trainer_params(
        "BERT Implicitly Labeled Reddit v2",
        hparams, RUNNING_DIR
    )

    trainer = generate_trainer(trainer_params)

    #
    trainer.fit(model, reddit_data_module)
    # trainer.test()


def wandb_sweep():
    import wandb

    config_defaults = {
        'CLASSES': ["suicidal-post"]
    }

    wandb.init(config=config_defaults)
    hparams = wandb.config

    main(hparams)


if __name__ == "__main__":
    RUNNING_DIR = r'C:\Code\NLP\ProfileLevel_SI_Classifier'
    hparams = Hyperparameters.from_file(
        path.join(RUNNING_DIR, 'reddit_hparams.json'))

    main(hparams, RUNNING_DIR=RUNNING_DIR)
