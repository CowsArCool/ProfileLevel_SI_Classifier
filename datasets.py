# Default Packages
import pandas as pd

# Pytorch/Lightning
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class RedditImplicit (Dataset):
    def __init__(self, reddit_df, tokenizer, max_example_len=512):
        # src is divided into input_ids, token_type_ids, and attention_mask
        self.src = tokenizer(
            reddit_df['text'].tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=max_example_len
        )

        self.trg = reddit_df['label'].replace(
            {'non-suicide': 0, 'suicide': 1}).tolist()

    @staticmethod
    def custom_vocab_preprocessing(df):
        return df['text']

    def __getitem__(self, idx):
        return [
            tuple([self.src['input_ids'][idx], self.src['attention_mask'][idx]]),
            torch.tensor(self.trg[idx])
        ]

    def __len__(self):
        assert len(self.src['input_ids']) == len(self.trg)
        return len(self.trg)

    def __str__(self):
        return f'RedditImplicit ({self.dataset_percent*100}% of full dataset)'


class RedditImplicitDataModule (pl.LightningDataModule):
    def __init__(
        self, data: pd.DataFrame,
        tokenizer, splits: list = [1],
        max_example_len: int = 512,
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 0
    ):

        super().__init__()

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_example_len = max_example_len
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.df_splits = list()
        datalen = len(data)
        for i, split_percent in enumerate(splits):
            prev_split = sum(splits[:i])

            self.df_splits.append(
                data[
                    int(prev_split*datalen):
                    int((prev_split * datalen) +
                        (split_percent*datalen)
                        )
                ]
            )

    def setup(self, stage=None):
        self.splits = [
            RedditImplicit(
                data,
                self.tokenizer,
                self.max_example_len
            )
            for data in self.df_splits
        ]

        if len(self.splits) <= 3:
            # complicated syntax making it possible to assign all three at once while padding
            # validset/testset if there arent enough splits to fill those values
            self.trainset, self.validset, self.testset = [
                split for split in self.splits] + [self.splits[-1]]*(3 - len(self.splits))

            self.datasets = {
                'train': self.trainset,
                'valid': self.validset,
                'test': self.testset
            }

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )


class TwitterLabeledSI (Dataset):
    def __init__(self, twitter_df, tokenizer, max_example_len=512):
        # src is divided into input_ids, token_type_ids, and attention_mask
        twitter_df.loc[:, 'tweet'] = twitter_df.loc[:, 'tweet'].apply(
            lambda x: x.strip().lower())
        self.src = tokenizer(
            twitter_df['tweet'].tolist(),
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=max_example_len
        )

        self.trg = twitter_df['label'].tolist()

    def __getitem__(self, idx):
        return [
            tuple([self.src['input_ids'][idx], self.src['attention_mask'][idx]]),
            torch.tensor(self.trg[idx])
        ]

    def __len__(self):
        assert len(self.src['input_ids']) == len(self.trg)
        return len(self.trg)

    def __str__(self):
        return f'LabeledTwitterSI ({self.dataset_percent*100}% of full dataset)'


class TwitterDataModule (pl.LightningDataModule):
    def __init__(
        self, data: pd.DataFrame,
        tokenizer, splits: list = [1],
        max_example_len: int = 512,
        shuffle: bool = True,
        batch_size: int = 32,
        num_workers: int = 0
    ):

        super().__init__()

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_example_len = max_example_len
        self.shuffle = shuffle
        self.num_workers = num_workers

        self.df_splits = list()
        datalen = len(data)
        for i, split_percent in enumerate(splits):
            prev_split = sum(splits[:i])

            self.df_splits.append(
                data[
                    int(prev_split*datalen):
                    int((prev_split * datalen) +
                        (split_percent*datalen)
                        )
                ]
            )

    def setup(self, stage=None):
        self.splits = [
            TwitterLabeledSI(
                data,
                self.tokenizer,
                self.max_example_len
            )
            for data in self.df_splits
        ]

        if len(self.splits) <= 3:
            # complicated syntax making it possible to assign all three at once while padding
            # validset/testset if there arent enough splits to fill those values
            self.trainset, self.validset, self.testset = [
                split for split in self.splits] + [self.splits[-1]]*(3 - len(self.splits))

            self.datasets = {
                'train': self.trainset,
                'valid': self.validset,
                'test': self.testset
            }

    def train_dataloader(self):
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.validset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
