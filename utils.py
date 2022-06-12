import os
import os.path as path
import json
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


import json
import warnings


class Hyperparameters:
    """ Serves to hold hyperparamaters with a number of
    methods to access them . Specifically works as a wandb
    config and can be replaced by a wandb sweep input

    """

    def __init__(self, ** kwargs):
        # Sets all kwargs to object attributes
        for key in kwargs:
            setattr(self, key, kwargs[key])

    @classmethod
    def from_file(cls, filename):
        with open(filename) as f:
            hparams = json.load(f)

        return cls(**hparams)

    def __getattr__(self, attr):
        # allows for None return when attribute cannot be found
        # raises a warning in this case
        try:
            return object.__getattribute__(self, attr)

        except AttributeError:
            if hasattr(self, 'silent'):
                if self.silent == True:
                    return None

            warnings.warn(
                "Hyperparameters has no attribute {}"
                    .format(attr)
            )

        return None

    def __getitem__(self, attr):
        return getattr(self, attr)


def prepare_wandb(project, display_name, hyperparamaters):
    return WandbLogger(
        name=display_name,
        project=project,
        config=hyperparamaters
    )


def generate_callbacks(display_name, hyperparamters, RUNNING_DIR):
    save_dir = path.join(
        RUNNING_DIR, 'model_checkpoints', display_name)
    os.makedirs(save_dir)
    early_stopping_callback = EarlyStopping(
        monitor='valid_loss', patience=hyperparamters['PATIENCE'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="valid_loss",
        mode="min"
    )

    return checkpoint_callback, early_stopping_callback


def generate_trainer_params(project, hyperparamaters, RUNNING_DIR):
    with open(path.join(RUNNING_DIR, 'words.txt')) as f:
        display_name = '-'.join(np.random.choice(
            (''.join(f.readlines()).split('\n')), size=2))
    print('Using display_name: {} for project: {}'.format(display_name, project))

    wandb_logger = prepare_wandb(project, display_name, hyperparamaters)
    callbacks = generate_callbacks(display_name, hyperparamaters, RUNNING_DIR)

    return {
        'logger': wandb_logger,
        'checkpoint_callback': callbacks[0],
        'callbacks': [callbacks[1]],
        'max_epochs': hyperparamaters['NUM_EPOCHS']
    }


def generate_trainer(trainer_params):
    # adds default paramaters
    return pl.Trainer(
        progress_bar_refresh_rate=30,
        gpus=1,
        **trainer_params
    )
