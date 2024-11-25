# Sklearn: pip install -U scikit-learn
# transformers: pip install transformers
# pytorch: https://pytorch.org/get-started/locally/
####################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
from models import Bert
from transformers import BertTokenizer, BertModel
import torch

import pandas as pd


class BertBaseMultiClassifier(BaseEstimator, ClassifierMixin):
    BERT_MODEL = 'bert-base-cased'

    def __init__(self, path='./model_multiclass.pt'):
        self.bert = BertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)
        self.model = Bert.MBERT(self.bert)
        self.model.load_state_dict(torch.load(
            path, map_location=torch.device('cpu'))['model_state_dict'])

    def predict(self, X, y=None):
        ids, token_type_ids, mask = self.process(X)
        out = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        return torch.nn.functional.softmax(out, dim=1)

    def process(self, X, y=None):
        ds = Bert.HC3DatasetForBert(self.tokenizer, X)
        dl = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        batch = next(iter(dl))

        return batch['ids'], batch['token_type_ids'], batch['mask']


if __name__ == "__main__":
    ds = pd.read_csv('Holdout_Final.csv')[:10]

    classifier = BertBaseMultiClassifier('model_multiclass.pt')
    print(classifier.predict(ds))
