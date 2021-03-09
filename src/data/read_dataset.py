# -*- coding: utf-8 -*-
"""
Script used to read datasets files.
"""
import pandas as pd
import logging
from typing import List, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split

project_dir = Path(__file__).resolve().parents[2]


def split_data(data, labels, train_frac, random_state=None) -> Tuple[str, str]:
    """
    param data: Data to be split
    param labels: labels to be used on stratify
    param train_frac: Ratio of train set to whole dataset

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'test': 1-train_frac

    Eg: passing train_frac=0.8 gives a 80% / 20% split
    """

    assert 0 <= train_frac <= 1, "Invalid training set fraction"

    train, test = train_test_split(data, train_size=train_frac, random_state=random_state, stratify=labels)
    return train, test


def balance_data(df):
    """
    param data: Dataframe to be balance

    Downsample majority class equal to the number of samples in the minority class
    """
    df_minority = df[df['accuracy'].apply(str) == 'False']
    df_majority = df[df['accuracy'].apply(str) == 'True']
    df_majority = df_majority.sample(len(df_minority), random_state=0)
    df = pd.concat([df_majority, df_minority])
    df = df.sample(frac=1, random_state=0)

    return df


def balance_data_language_pair(df: pd.DataFrame) -> pd.DataFrame:
    """
    param df: Dataframe to be balance

    Downsample majority class equal to the number of samples in the minority class filtering by language pair
    """
    appended_df = []
    languages = df['language_pair'].unique()
    for language_pair in languages:
        appended_df.append(balance_data(df[df['language_pair'] == language_pair]))
    df = pd.concat(appended_df)

    return df


def get_data_validation() -> Tuple[List[str], List[str]]:
    """Reads Validation dataset."""
    path = project_dir / 'data' / 'processed' / 'validation'
    if path.exists():
        try:
            train = pd.read_csv(path / 'train.csv', delimiter=",",
                                header=0, encoding='utf-8', engine='python')
            dev = pd.read_csv(path / 'dev.csv', delimiter=",",
                              header=0, encoding='utf-8', engine='python')
            test = pd.read_csv(path / 'test.csv', delimiter=",",
                                header=0, encoding='utf-8', engine='python')
        except pd.errors.EmptyDataError:
            print("file is empty and has been skipped.")
        return train, dev, test


class DatasetReader:
    """Handles dataset reading"""

    def __init__(self, language_pairs: str):
        self._language_pairs = language_pairs

    def get_data_validation(self) -> Tuple[List[str], List[str], List[str]]:
        """Reads Validation dataset."""
        appended_src = []
        appended_tgt = []
        appended_tags = []
        for language_pair in self._language_pairs:
            logging.info(f'processing {language_pair}')
            path = project_dir / 'data' / 'processed' / 'validation' / language_pair
            if path.exists():
                try:
                    src_file = pd.read_csv(path / 'source.txt', delimiter="\n",
                                           header=None, encoding='utf-8', engine='python')
                    src_file.columns = ['source']
                    appended_src.append(src_file)
                    trg_file = pd.read_csv(path / 'target.txt', delimiter="\n",
                                           header=None, encoding='utf-8', engine='python')
                    trg_file.columns = ['target']
                    appended_tgt.append(trg_file)
                    tags_file = pd.read_csv(path / 'tags.txt', delimiter="\n",
                                            header=None, encoding='utf-8', engine='python')
                    tags_file.columns = ['accuracy']
                    tags_file = tags_file['accuracy'].apply(str)
                    appended_tags.append(tags_file)

                    assert len(src_file) == len(trg_file) == len(tags_file)
                except pd.errors.EmptyDataError:
                    print(language_pair + " is empty and has been skipped.")
            else:
                logging.info(f'directory does not exist {path}')

        df_src = pd.concat(appended_src)
        df_tgt = pd.concat(appended_tgt)
        df_tags = pd.concat(appended_tags)
        df_validation = pd.concat([df_src, df_tgt, df_tags], axis=1)
        return df_validation

    def get_data_inspection(self) -> Tuple[List[str], List[str], List[str]]:
        """Reads Inspection dataset"""
        path = project_dir / 'data' / 'processed' / 'inspection' / self._language_pairs
        if path.exists():
            try:
                src_file = pd.read_csv(path / 'source.txt', delimiter="\n",
                                       header=None, encoding='utf-8', engine='python')
                trg_file = pd.read_csv(path / 'target.txt', delimiter="\n",
                                       header=None, encoding='utf-8', engine='python')
                gold_file = pd.read_csv(path / 'gold.txt', delimiter="\n",
                                        header=None, encoding='utf-8', engine='python')

                assert len(src_file) == len(trg_file) == len(gold_file)

                src_sentences = src_file[0].tolist()
                trg_sentences = trg_file[0].tolist()
                gold_sentences = gold_file[0].tolist()
                return src_sentences, trg_sentences, gold_sentences
            except pd.errors.EmptyDataError:
                print(self._language_pairs + " is empty and has been skipped.")
        else:
            logging.info(f'directory does not exist {path}')