import os
import argparse
import logging
import math
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, LoggingHandler
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
from src.data.read_dataset import get_data_validation
from sentence_transformers.util import pytorch_cos_sim
from sklearn import metrics
from sklearn.preprocessing import minmax_scale
from src.models.predict_model import SentenceTransformerPredict

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
label2int = {'False': 0, 'True': 1}

project_dir = Path(__file__).resolve().parents[2]


class SentenceTransformerFineTuning:
    """
    Provides a fine-tuned sentence transformer model based on pre-trained models
    Usage:
    python -m src.models.train_model
    OR
    python -m src.models.train_model en-it
    """

    def __init__(self, model_name="distiluse-base-multilingual-cased-v2"):
        self._model_name = model_name

    def validation(self, num_folds, cross_validation) -> None:
        """Validate Sentence Transformer models using cross-validation and AUC Score
        Args:
            num_folds (int): folds number
            cross_validation (boolean): cross-validation?
        """
        # get data
        df_train, df_dev, df_test = get_data_validation()
        stf = SentenceTransformerFineTuning()
        evaluation_custom = []
        evaluation_baseline = []

        if cross_validation:
            # prepare the cross-validation procedure
            labels = pd.concat([df_train['accuracy'].apply(str)], axis=1)
            skf = StratifiedKFold(n_splits=num_folds, random_state=None, shuffle=False)
            df = df_train
            for fold, (train_index, val_index) in enumerate(skf.split(df_train, labels)):
                df_train = df.loc[train_index]
                df_val = df.loc[val_index]

                # fit
                model = stf.fit(df_train, '')
                model = SentenceTransformerPredict(str(model))

                # predict
                df_val["custom"] = df_val.apply(lambda row: model.predict(row.source, row.target), axis=1)
                df_val['custom'] = minmax_scale(df_val["custom"])

                # evaluate
                evaluation_custom.append(stf.evaluate(df_val['accuracy'], df_val['custom'], 'roc_auc'))

            # report performance
            print('Evaluation: %.3f (%.3f)' % (np.mean(evaluation_custom), np.std(evaluation_custom)))
        else:
            # fit
            model = stf.fit(df_train, df_dev)
            model_custom = SentenceTransformerPredict(str(model))
            model_baseline = SentenceTransformerPredict(str(self._model_name))

            # predict
            df_test["custom"] = df_test.apply(lambda row: model_custom.predict(row.source, row.target), axis=1)
            df_test['custom'] = minmax_scale(df_test["custom"])

            df_test["baseline"] = df_test.apply(lambda row: model_baseline.predict(row.source, row.target), axis=1)
            df_test['baseline'] = minmax_scale(df_test["baseline"])

            # evaluate
            evaluation_custom.append(stf.evaluate(df_test['accuracy'], df_test['custom'], 'roc_auc'))
            evaluation_baseline.append(stf.evaluate(df_test['accuracy'], df_test['baseline'], 'roc_auc'))

            # report performance
            print('Evaluation Custom Model: ', evaluation_custom)
            print('Evaluation Baseline Model: ', evaluation_baseline)

    def format_data(self, train, dev) -> Tuple[str, str]:
        """Convert data in a SentenceTransformer format
        Args:
            train (dataframe): training set
            dev (dataframe): development set
        Returns:
            train and dev samples
        """
        train_samples = []
        dev_samples = []
        if isinstance(train, pd.DataFrame):
            for src_sentence, trg_sentence, accuracy in zip(train['source'], train['target'], train['accuracy'].apply(str)):
                label_id = label2int[accuracy]
                train_samples.append(InputExample(texts=[src_sentence, trg_sentence], label=label_id))
        if isinstance(dev, pd.DataFrame):
            for src_sentence, trg_sentence, accuracy in zip(dev['source'], dev['target'], dev['accuracy'].apply(str)):
                label_id = label2int[accuracy]
                dev_samples.append(InputExample(texts=[src_sentence, trg_sentence], label=label_id))
        return train_samples, dev_samples

    def fit(self, train, dev) -> str:
        """Fine-tune a new SentenceTransformer model
        Args:
            train (dataframe): training set
            dev (dataframe): development set
        Returns:
            SentenceTransformer model: new trained model
        """
        # parameters
        train_batch_size = 16
        language_folder = 'all_languages'
        model_save_path = project_dir / 'models' / language_folder / \
                          str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(model_save_path)
        num_epochs = 1
        model = SentenceTransformer(self._model_name)
        stf = SentenceTransformerFineTuning()

        # load data
        train_samples, dev_samples = stf.format_data(train, dev)
        train_dataset = SentencesDataset(train_samples, model=model)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
        train_loss = losses.ContrastiveLoss(model=model, margin=0.1)

        # train
        if dev_samples:
            dev_evaluator = BinaryClassificationEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='binary-dev')
        else:
            dev_evaluator = None

        warmup_steps = math.ceil(
            len(train_dataset) * num_epochs / train_batch_size * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        model.fit(train_objectives=[(train_dataloader, train_loss)],
                  evaluator=dev_evaluator,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path
                  )

        model.save(model_save_path)
        return model_save_path

    def evaluate(self, accuracy, scores, metric) -> float:
        """Evaluate a SentenceTransformer model
        Args:
            scores (float): predicted cosine similarity scores
        Returns:
            metric (float): ROC AUC score
        """
        accuracy = accuracy.apply(str).values.tolist()
        y = np.array([label2int[label] for label in accuracy])
        pred_model = np.array([float(score) for score in scores])
        if metric == 'roc_auc':
            fpr, tpr, thresholds = metrics.roc_curve(y, pred_model, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)

        return roc_auc


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to fine-tune sentence transformer model based on pre-trained models.''')

    parser.add_argument('language_pairs',
                        nargs='*',
                        help='{en-it, en-ru} choose English-Italian or English-Russian testset')

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = parser.parse_args()

    language_pairs = args.language_pairs
    if len(language_pairs) == 0:
        path_files = project_dir.joinpath('data', 'processed', 'validation')
        language_pairs = [path.name for path in path_files.glob('*') if path.is_dir()]
    SentenceTransformerFineTuning().validation(10, False)
