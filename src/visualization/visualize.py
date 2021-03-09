# -*- coding: utf-8 -*-
import torch
import logging
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer, util, LoggingHandler
from src.data.read_dataset import get_data_validation

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

project_dir = Path(__file__).resolve().parents[2]


class SentenceEmbeddingVisualize:
    """
    Provides visualizations using sentence embeddings models
    Usage:
    python -m src.visualization.visualize
    """

    def __init__(self, language_pairs: str, model_name="distiluse-base-multilingual-cased"):
        self._model_name = model_name
        self._language_pairs = language_pairs

    def visualize_tensorboard(self) -> None:
        logging.info("Create the tensorboard files (tensors and metadata)")

        model = SentenceTransformer(self._model_name)

        train, dev, test = get_data_validation()
        embeddings = []
        cosine_distance = []
        metadata = []

        for accuracy, src_sentence, trg_sentence, score in zip(test['accuracy'], test['source'], test['target'], test['score_baseline']):
            # encode the source sentence
            src_tensor = model.encode(src_sentence, convert_to_tensor=True).data.cpu().numpy()
            embeddings.append(src_tensor)

            # encode the target sentence
            trg_tensor = model.encode(trg_sentence, convert_to_tensor=True).data.cpu().numpy()
            embeddings.append(trg_tensor)

            # calculate the cosine distance between source and target
            cosine_distance.append(util.pytorch_cos_sim(model.encode(src_sentence, convert_to_tensor=True), model.encode(trg_sentence, convert_to_tensor=True))[0])
            cosine_distance.append([score])

            # add ground truth (accuracy) to the metadata
            metadata.append(src_sentence + '\t' + str(accuracy))
            metadata.append(trg_sentence + '\t' + str(accuracy))

        embeddings_tensor = torch.FloatTensor(embeddings)

        # create the SummaryWriter object
        writer = SummaryWriter()
        writer.add_embedding(embeddings_tensor, metadata=metadata)
        writer.close()


if __name__ == '__main__':
    # Parser descriptors
    parser = argparse.ArgumentParser(
        description='''Script used to visualize sentence embeddings.''')

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
    SentenceEmbeddingVisualize(language_pairs).visualize_tensorboard()