# -*- coding: utf-8 -*-
"""
Script used to read external files in order to generate training, development and test sets.
"""

import os
import argparse
import logging
import pandas as pd
import json
import csv
from pathlib import Path
from typing import List, Tuple
from src.data.text_preprocessing import TextPreprocessing
from src.data.read_dataset import balance_data, balance_data_language_pair, split_data

project_dir = Path(__file__).resolve().parents[2]


def convert_parallel_to_tsv(language_pair:str ) -> None:
    """Take source and target and create a single parallel file, with \t as a separator."""
    path = project_dir / 'data' / 'processed' / 'inspection' / language_pair
    if path.exists():
        src = pd.read_csv(path / 'source.txt', delimiter="\n",
                          header=None, encoding='utf-8', engine='python')
        tgt = pd.read_csv(path / 'target.txt', delimiter="\n",
                          header=None, encoding='utf-8', engine='python')
        assert len(src) == len(tgt)
        df = pd.DataFrame(src + '\t' + tgt)
        output = path / 'parallel.tsv'
        df.to_csv(output, sep='\t',
                  encoding='utf-8', index=False, header=None)
        logging.info(f'written to {output}')


def create_inspection_files(language_pair: str) -> None:
    """Converts each column from the quality inspection spreadsheet to separate .txt files"""
    path = project_dir / 'data' / 'raw' / 'inspection' / language_pair
    output = project_dir / 'data' / 'processed' / 'inspection' / language_pair
    if path.exists():
        if not output.exists():
            os.mkdir(output)
        for entry in path.glob('*.csv'):
            df = pd.read_csv(entry, delimiter=";", header=0,
                                encoding='utf-8', engine='python')
            logging.info(f'reading {entry}')
            with open(output / 'error_category.txt', 'a+', encoding='utf8') as f_error_category, \
                    open(output / 'error_type.txt', 'a+', encoding='utf8') as f_error_type, \
                    open(output / 'severity.txt', 'a+', encoding='utf8') as f_severity, \
                    open(output / 'source.txt', 'a+', encoding='utf8') as f_source, \
                    open(output / 'target.txt', 'a+', encoding='utf8') as f_target, \
                    open(output / 'gold.txt', 'a+', encoding='utf8') as f_gold:
                for error_category, error_type, severity, source, target, gold in df.values.tolist():
                    if all(map(validate_columns,
                                (error_category, error_type, severity, source, target, gold))):
                        f_error_category.write(
                            (str(error_category.strip())) + '\n')
                        f_error_type.write(
                            (str(error_type.strip())) + '\n')
                        f_severity.write(
                            (str(severity.strip())) + '\n')
                        f_source.write(
                            (str(source.strip())) + '\n')
                        f_target.write(
                            (str(target.strip())) + '\n')
                        f_gold.write(
                            (str(gold.strip())) + '\n')
            logging.info(f'finished to process {entry}')
        else:
            logging.info(f'directory is already created {output}')


def create_validation_files(language_pair: str) -> None:
    """Converts each column from the quality validation spreadsheet to separate .txt files"""
    path = project_dir / 'data' / 'raw' / 'validation' / language_pair
    output = project_dir / 'data' / 'processed' / 'validation' / language_pair
    data = []
    if path.exists():
        if not output.exists():
            os.mkdir(output)
        for entry in path.glob('*.xlsx'):
            logging.info(f'reading {entry}')
            file = pd.read_excel(entry, header=0)
            df = pd.DataFrame(file)
            columns = df.values.tolist()

            with open(output / 'project.txt', 'a+', encoding='utf8') as f_project, \
                    open(output / 'source.txt', 'a+', encoding='utf8') as f_source, \
                    open(output / 'target.txt', 'a+', encoding='utf8') as f_target, \
                    open(output / 'tags.txt', 'a+', encoding='utf8') as f_tags:
                for job_id, hit_id, title, language_requirement, result, input_for_ui_controls in columns:
                    content = json.loads(input_for_ui_controls.strip())
                    labels = json.loads(result.strip())
                    if 'metadata' in labels:
                        labels = labels['metadata']
                    source_column, target_column, label_column = define_columns(content, labels)
                    if validate_columns(source_column) and validate_columns(target_column) and validate_columns(label_column):
                        if (str(labels[label_column]).lower() == 'false') or (str(labels[label_column]).lower() == '0'):
                            tags = 'false'
                            f_tags.write('false\n')
                        elif (str(labels[label_column]).lower() == 'true') or (str(labels[label_column]).lower() == '1'):
                            tags = 'true'
                            f_tags.write('true\n')
                        else:
                            continue
                        f_project.write(
                            str(job_id) + ' ' + str(hit_id) + ' ' + str(language_requirement) + '\n')
                        source = TextPreprocessing(str(content[source_column])).remove_html()
                        target = TextPreprocessing(str(content[target_column])).remove_html()
                        f_source.write(source + '\n')
                        f_target.write(target + '\n')

                        # save the data in a format to be later exported to a dataframe
                        data_instance = {}
                        data_instance['language_pair'] = language_pair
                        data_instance['source'] = source
                        data_instance['target'] = target
                        data_instance['accuracy'] = tags
                        data_instance['jobid'] = ''.join(filter(str.isdigit, entry.stem))
                        if str(entry).endswith('agreement.xlsx'):
                            data_instance['agreement'] = True
                        else:
                            data_instance['agreement'] = False
                        data.append(data_instance)
            logging.info(f'finished to process {entry}')
    else:
        logging.info(f'directory is already created {output}')
    return pd.DataFrame().from_dict(data)


def create_train_test_sets(dataset_file: str, train_frac: float, balanced: bool, balanced_language: bool, testset_csv: str, job_ids: [str]) -> None:
    """
    Create training and test sets with the same distribution for language pair and accuracy (class), in order to train and evaluate trained models

    param dataset_file: Data to be split
    param train_frac: Ratio of train set to whole dataset
    param balanced: Downsample majority class equal to the number of samples in the minority class
    param balanced_language: Downsample majority class equal to the number of samples in the minority class filtering by language pair
    param testset_csv: External test set file (csv format)
    param job_ids: List of job ids that will be part of the test set

    Randomly split dataset, based on these ratios:
        'train': train_frac
        'test': 1-train_frac

    Eg: passing train_frac=0.8 gives a 80% / 20% split
    """
    path = project_dir / 'data' / 'processed' / 'validation'
    df = pd.read_csv(dataset_file, delimiter=",",
                           header=0, encoding='utf-8', engine='python')
    test = pd.DataFrame()

    # Is there a test set csv available?
    if testset_csv != '':
        df = pd.DataFrame(df, columns=['language_pair', 'source', 'target', 'accuracy'])
        testset = pd.read_csv(path / testset_csv, delimiter=",",
                              usecols=['language_pair', 'source', 'target', 'accuracy'],
                              header=0, encoding='utf-8', engine='python')
        if not testset.empty:
            test = testset
            df = pd.concat([df, testset, testset]).drop_duplicates(keep=False)

    # Are there jobs that will be part of the test set?
    elif job_ids:
        appended_test = []
        for job_id in job_ids:
            appended_test.append(df.loc[df['jobid'] == job_id])
            df = df.loc[df['jobid'] != job_id]
        test = pd.concat(appended_test)
        df = pd.concat([df, test, test]).drop_duplicates(keep=False)

    # Balance by class
    if balanced:
        df = balance_data(df)
    # Balance by class filtering by language pair
    elif balanced_language:
        df = balance_data_language_pair(df)

    # Get labels
    labels = pd.concat([df['language_pair'], df['accuracy']], axis=1)

    # Split data
    train, dev = split_data(df, labels, train_frac)

    # Resume
    logging.info('\ntrain-------------------------------------------------------------')
    logging.info(train.shape)
    logging.info('label     %')
    logging.info(f" {round(train.groupby('language_pair')['source'].count() * 100 / train.shape[0], 2)}")
    logging.info(f" {round(train.groupby('accuracy')['source'].count() * 100 / train.shape[0], 2)}")

    logging.info('\ndev-------------------------------------------------------------')
    logging.info(dev.shape)
    logging.info('label     %')
    logging.info(f" {round(dev.groupby('language_pair')['source'].count() * 100 / dev.shape[0], 2)}")
    logging.info(f" {round(dev.groupby('accuracy')['source'].count() * 100 / dev.shape[0], 2)}")

    # Save files
    train.to_csv(path / 'train.csv', index=False)
    dev.to_csv(path / 'dev.csv', index=False)
    test.to_csv(path / 'test.csv', index=False)


def validate_columns(column) -> bool:
    """Validates that the columns are not empty and are a string"""
    return isinstance(column, str) and column.strip() != "" and column.strip() != "null"


def define_columns(content, labels) -> Tuple[str, str, str]:
    """Defines column content based on different templates"""
    source_column = ''
    target_column = ''
    label_column = ''
    if 'source' in content:
        source_column = 'source'
    elif 'origintext' in content:
        source_column = 'origintext'

    if 'target' in content:
        target_column = 'target'
    elif 'translatedtext' in content:
        target_column = 'translatedtext'

    if 'accuracy' in labels:
        label_column = 'accuracy'

    return source_column, target_column, label_column


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to read external files in order to generate training, development and test sets.''')

    parser.add_argument('language_pairs',
                        nargs='*',
                        help='{en-it, en-ru} choose English-Italian or English-Russian testset')

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parser.parse_args()

    language_pairs = args.language_pairs
    if len(language_pairs) == 0:
        language_pairs = [path.name for path in
                          project_dir.joinpath('data', 'raw', 'inspection').glob('*') if path.is_dir()]
        language_pairs = language_pairs + [path.name for path in
                                           project_dir.joinpath('data', 'raw', 'validation').glob('*') if path.is_dir()]
        # remove duplicated entries
        language_pairs = set(language_pairs)

    df = pd.DataFrame()
    df_lang = []
    for language_pair in language_pairs:
        logging.info(f'processing {language_pair}')
        create_inspection_files(language_pair)
        convert_parallel_to_tsv(language_pair)
        df_lang.append(create_validation_files(language_pair))
    df = pd.concat(df_lang)

    # output a single csv with the entire validation dataset
    output_path = project_dir / 'data' / 'processed' / 'validation' / 'dataset.csv'
    df.to_csv(output_path, index=False)

    create_train_test_sets(output_path, 0.9, True, True, '', [4934, 5113, 5479, 5526, 6708, 6741])