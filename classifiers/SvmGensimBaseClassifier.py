# Specify what to pip install
# - scikit-learn
# - nltk
# - gensim
# - tqdm

# Specify your imports
from tqdm import tqdm
import gensim.downloader
from collections import defaultdict
from sklearn.base import BaseEstimator, ClassifierMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import pickle
import nltk
import numpy as np
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


class SvmGensimBaseClassifier(BaseEstimator, ClassifierMixin):

    vectorizers = {
        "glove": 'glove-wiki-gigaword-300',
        "w2v": 'word2vec-google-news-300',
        "fasttext": "fasttext-wiki-news-subwords-300"
    }

    def __init__(self, path_model: str, chosen_vectorizer: str):
        # 'svm_rbf_model_glove.pkl'
        self.model = pickle.load(open(path_model, 'rb'))
        self.vectorizer = self.initialize_vectorizer(chosen_vectorizer)

    def predict(self, X, y=None):
        processed_X = self.process(X)
        return self.model.predict(processed_X)

    def process(self, X, y=None):
        processed_X = self.input_preprocess(X["response"])
        return processed_X

    def initialize_vectorizer(self, chosen_vectorizer):
        downloaded = False
        max_failures = 5
        glove = None

        while not downloaded and max_failures > 0:
            try:
                glove = gensim.downloader.load(
                    self.vectorizers[chosen_vectorizer])
                downloaded = True
            except:
                print("download failed, retrying...")
                max_failures -= 1
        if glove == None:
            raise Exception("Vectorizer failed to initialize")
        else:
            return glove

    # function to vectorize tokenized text
    def generateVectors(self, X):
        results = np.zeros((len(X), self.vectorizer.vector_size))
        for i, x in enumerate(X):
            vector = np.zeros(self.vectorizer.vector_size)
            for word in x:
                if word in self.vectorizer:
                    vector = vector + self.vectorizer[word]
            results[i] = vector
        return results

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
        w2v_vectors = self.generateVectors(text_arr)

        return w2v_vectors
