# -*- coding: utf-8 -*-
"""
Script used to evaluate a pretrained sentence transformer model.
"""
import os
import logging
import argparse
from pathlib import Path
from src.data.read_dataset import DatasetReader
from sentence_transformers import SentenceTransformer, util, LoggingHandler, models, evaluation, losses

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

project_dir = Path(__file__).resolve().parents[2]


def retrieval_accuracy(language_pair: str, model: str) -> None:
    """Check if the sentence pair have the shortest cosine distance between source and target."""
    inference_batch_size = 32
    path = project_dir / 'data' / 'processed' / 'inspection' / language_pair / 'parallel.tsv'
    src_sentences, trg_sentences, gold_sentences = DatasetReader(language_pair).get_data_inspection()
    logging.info(str(len(src_sentences)) + " sentence pairs")
    dev_trans_acc = evaluation.TranslationEvaluator(src_sentences, trg_sentences, name=os.path.basename(
        path), batch_size=inference_batch_size, print_wrong_matches=False)
    dev_trans_acc(model)


def ranking_accuracy(language_pair: str, model: str) -> None:
    """Check if the cosine distance between the source and the target is lower than the source and the gold target."""
    scores = []
    scores_gold = []
    src_sentences, trg_sentences, gold_sentences = DatasetReader(language_pair).get_data_inspection()
    for src_sentence, trg_sentence, gold_sentence in zip(src_sentences, trg_sentences, gold_sentences):
        source_embedding = model.encode(src_sentence, convert_to_tensor=True)
        target_embedding = model.encode(trg_sentence, convert_to_tensor=True)
        gold_embedding = model.encode(gold_sentence, convert_to_tensor=True)
        scores.append(util.pytorch_cos_sim(
            source_embedding, target_embedding)[0])
        scores_gold.append(util.pytorch_cos_sim(
            source_embedding, gold_embedding)[0])

    results = zip(range(len(scores)), scores, scores_gold)
    results = sorted(results, key=lambda x: x[1], reverse=False)

    correct_score = 0
    for idx, score, score_gold in results:
        if score_gold >= score:
            correct_score += 1

        print('Source: ', src_sentences[idx])
        print('Target: ', trg_sentences[idx])
        print('Gold: ', gold_sentences[idx])
        print("(Score: %.4f)" % (score))
        print("(Score Gold: %.4f)" % (score_gold))
        print("*******************************************************************************************")

    acc_score = correct_score / len(results)
    logging.info(str(len(results)) + " sentence pairs")
    logging.info("Accuracy: {:.2f}".format(acc_score * 100))


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to evaluate trained models.''')

    parser.add_argument('language_pair',
                        choices=['en-ar', 'en-it', 'en-jp',
                                 'en-ko', 'en-ru', 'ru-fr'],
                        help='{en-it, en-ru} choose English-Italian or English-Russian testset')
    parser.add_argument('--model',
                        type=str,
                        default='distiluse-base-multilingual-cased',
                        help='pretrained sentence transformer model, e.g. distiluse-base-multilingual-cased')

    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    retrieval_accuracy(args.language_pair, model)
    ranking_accuracy(args.language_pair, model)
