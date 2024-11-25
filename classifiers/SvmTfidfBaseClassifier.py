from tqdm import tqdm
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class SvmTfidfBaseClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, path_model: str, path_vectorizer: str):
        self.model = pickle.load(open(path_model, 'rb'))
        self.vectorizer = pickle.load(
            open(path_vectorizer, 'rb'))

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return self.model.predict(processed_X)

    def process(self, X, y=None):
        X_processed = self.input_preprocess(X["response"])
        return X_processed

    def input_preprocess(self, text_arr: list) -> list:
        # change all the words to lower case
        text_arr = [text.lower() for text in text_arr]

        # tokenize the text
        text_arr = [word_tokenize(text) for text in text_arr]

        # remove stopwords, non-numeric and lemmatize the words
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        word_Lemmatized = WordNetLemmatizer()

        for index, entry in tqdm(enumerate(text_arr)):
            Final_words = []
            for word, tag in pos_tag(entry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(
                        word, tag_map[tag[0]])
                    Final_words.append(word_Final)
            text_arr[index] = str(Final_words)

        # vectorize the words
        X_Tfidf = self.vectorizer.transform(text_arr)

        return X_Tfidf


class SvmTfidfBaseClassifierV2(BaseEstimator, ClassifierMixin):
    def __init__(self, path_model: str, path_vectorizer: str):
        self.model = pickle.load(open(path_model, 'rb'))
        self.vectorizer = pickle.load(
            open(path_vectorizer, 'rb'))

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return self.model.predict(processed_X)

    def process(self, X, y=None):
        X_processed = self.vectorizer.transform(X['response'])
        return X_processed

    def predict_proba(self, X, y=None):
        processed_X = self.process(X)
        return self.model.predict_proba(processed_X)
