# Default Packages
import os
import os.path as path
import pandas as pd

# Tokenization
from transformers import BertTokenizer

# Internal Packages
from models import SuicideClassifier
from utils import *
from datasets import TwitterDataModule


def read_twitterSI(data_path, n_examples=None):
    df = pd.read_csv(data_path)

    if n_examples is None:
        n_examples = len(df)
    df = df.sample(n_examples)

    df.reset_index(inplace=True, drop=True)

    return df


RUNNING_DIR = r'C:\Code\NLP\ProfileLevel_SI_Classifier'
datasets_dir = path.join(RUNNING_DIR, 'Datasets')

MODEL_SAVE_NAME = 'flour-horse'


def main(hparams, RUNNING_DIR=os.path.dirname(path.realpath(__file__))):
    twitter_df = read_twitterSI(
        path.join(datasets_dir, 'Origional Suicidal Tweets.csv'),
        n_examples=hparams.N_EXAMPLES
    )

    tokenizer = BertTokenizer.from_pretrained(hparams.BERT_MODEL)
    twitter_data_module = TwitterDataModule(
        twitter_df,
        tokenizer,
        splits=[0.7, 0.3],
        max_example_len=hparams.MAX_EXAMPLE_LEN,
        shuffle=True,
        batch_size=hparams.BATCH_SIZE,
    )

    twitter_training_steps = len(
        twitter_df)//hparams.BATCH_SIZE*hparams.NUM_EPOCHS
    loaded_model = SuicideClassifier.load_from_checkpoint(
        checkpoint_path=path.join(
            RUNNING_DIR, 'model_checkpoints',
            f'{MODEL_SAVE_NAME}', 'best-checkpoint.ckpt'),
        training_steps=twitter_training_steps,
        warmup_steps=twitter_training_steps/5,
        lr=hparams.LEARNING_RATE,
        metrics=['ROC', 'binary_report']
    )

    twitter_trainer_params = generate_trainer_params(
        "TwitterSI Classification",
        hparams, RUNNING_DIR
    )

    twitter_trainer = generate_trainer(twitter_trainer_params)

    twitter_trainer.fit(loaded_model, twitter_data_module)
    # twitter_trainer.test()


def wandb_sweep():
    import wandb

    config_defaults = {
        'CLASSES': ["suicidal-tweet"]
    }

    wandb.init(config=config_defaults)
    hparams = wandb.config

    main(hparams)


if __name__ == '__main__':
    RUNNING_DIR = r'C:\Code\NLP\ProfileLevel_SI_Classifier'
    hparams = Hyperparameters.from_file(
        path.join(RUNNING_DIR, 'twitter_hparams.json'))

    main(hparams, RUNNING_DIR=RUNNING_DIR)
