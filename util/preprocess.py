import string

class Preprocess:

    def __init__(self):
        self.text = None

    def preprocess(self, text):
        # konversi ke non kapital
        self.text = text.lower()
        # hilangkan tanda baca
        tandabaca = tuple(string.punctuation)
        self.text = ''.join(ch for ch in self.text if ch not in tandabaca)
        return self.text
