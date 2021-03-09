import numpy as np
from sentence_transformers import SentenceTransformer, evaluation, LoggingHandler
from sentence_transformers.util import pytorch_cos_sim


class CrosslingualCosineSimilarity:
    """Provides a cross lingual similarity model based on sentence transformers and pre-trained models"""

    def __init__(self, model_name="distiluse-base-multilingual-cased-v2"):
        self._model = SentenceTransformer(model_name)

    def predict(self, source_sentence: str, target_sentence: str) -> np.float:
        """Predict the consine similarity between two sentences using a SentenceTransformer model

        Args:
            source_sentence (str): Source sentence
            target_sentence (str): Target Sence

        Returns:
            np.float: consine similarity between the two sentences ranging between 0 (less similar) and 1 (more similar)
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