import numpy as np
import logging
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers.util import pytorch_cos_sim

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.ERROR,
                    handlers=[LoggingHandler()])


class SentenceTransformerPredict:
    """Provides a prediction based on sentence transformers and pre-trained models"""

    def __init__(self, model):
        self._model = SentenceTransformer(model)

    def predict(self, source_sentence: str, target_sentence: str) -> np.float:
        """Predict the cosine similarity between two sentences using a SentenceTransformer model

        Args:
            source_sentence (str): Source sentence
            target_sentence (str): Target sentence

        Returns:
            np.float: Cosine similarity between the two sentences ranging between 0 (less distance) and 1 (more distance)
        """
        if self._model:
            source_embedding = self._model.encode(
                source_sentence, convert_to_tensor=True
            )
            target_embedding = self._model.encode(
                target_sentence, convert_to_tensor=True
            )
            score = pytorch_cos_sim(source_embedding, target_embedding)
            return score.cpu().numpy()[0][0]