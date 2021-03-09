# -*- coding: utf-8 -*-
"""
Script that connects the ModelFront API and calculates the prediction risk for specific sentence pairs.
"""
import requests
import json
import logging
from typing import List
from pathlib import Path
from src.data.read_dataset import get_data_validation

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

project_dir = Path(__file__).resolve().parents[2]

MODELFRONT_TOKEN = 'de1bc14fc4ddc46ae0b2818011cdf9b7fbd9f2247ed340a190c352995e00d2eb'
output = project_dir / 'data' / 'processed' / 'validation'


def get_translation_risk(test) -> List[str]:
    appended_score = []
    for language_pair, src_sentence, trg_sentence in zip(test['language_pair'], test['source'], test['target']):
        logging.info(f'written to {language_pair}')
        try:
            language = language_pair.split('-')

            # Japanese prefix setting
            language[0] = 'ja' if language[0] == 'jp' else language[0]
            language[1] = 'ja' if language[1] == 'jp' else language[1]

            # Data format
            data = {"rows": [{"original": src_sentence, "translation": trg_sentence}]}
            data = json.dumps(data)

            # API Request
            r = requests.post('https://api.modelfront.com/v1/predict?sl=' + language[0] + '&tl=' + language[
                1] + '&token=' + MODELFRONT_TOKEN, data=data.encode('utf-8'))
            res = json.loads(r.text)

            # Receives the risk prediction
            score = res['rows'][0]['risk']

            # Converts risk to a quality score
            appended_score.append(1 - (score / 100))
        except Exception as e:
            logging.info('Failed to access the API: ' + str(res))

    # Adds the new column to the test set
    return appended_score


if __name__ == '__main__':
    train, dev, test = get_data_validation()
    get_translation_risk(test)