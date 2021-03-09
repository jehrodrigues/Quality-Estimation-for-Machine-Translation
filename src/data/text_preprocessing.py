from bs4 import BeautifulSoup


class TextPreprocessing(object):
    """
    Handles text pre-processing
    """

    def __init__(self, sentence: str):
        self._sentence = sentence

    def remove_html(self) -> str:
        """Take a sentence and remove the html tags."""
        soup = BeautifulSoup(self._sentence, 'html.parser')
        clean_sentence = soup.get_text()
        clean_sentence = clean_sentence.replace("\"","")
        return clean_sentence
