# Sklearn: pip install -U scikit-learn
# transformers: pip install transformers
# pytorch: https://pytorch.org/get-started/locally/
####################################################################
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
sys.path.append('../')
from models.Bert import BERT, HC3DatasetForBert
from transformers import BertTokenizer, BertModel
import torch

import pandas as pd
class BertBaseClassifier(BaseEstimator, ClassifierMixin):
    BERT_MODEL = 'bert-base-cased'

    def __init__(self, path='./model.pt'):
        self.bert = BertModel.from_pretrained(self.BERT_MODEL)
        self.tokenizer = BertTokenizer.from_pretrained(self.BERT_MODEL)
        self.model = BERT(self.bert)
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(path)['model_state_dict'])
        else:
            self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model_state_dict'])

    def predict(self, X, y=None):
        '''
        Returns logits where >=0 is predicted as 1 (AI generated text) and 0 (Human text) otherwise.
        '''
        ids, token_type_ids, mask = self.process(X)
        return self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    def process(self, X, y=None):
        X = X[["response"]]
        ds = HC3DatasetForBert(self.tokenizer, X)
        dl = torch.utils.data.DataLoader(ds, batch_size=len(ds))
        batch = next(iter(dl))
        return batch['ids'], batch['token_type_ids'], batch['mask']