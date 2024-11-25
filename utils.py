from sklearn.pipeline import Pipeline
from classifiers.BertBaseClassifier import BertBaseClassifier
from classifiers.BertBaseMultiClassifier import BertBaseMultiClassifier
from classifiers.KerasModelClassifier import CNNGLTRClassifier
from classifiers.SvmTfidfBaseClassifier import SvmTfidfBaseClassifierV2
import pandas as pd
import numpy as np
from input_preprocess.GLTRTransformer import LM, GLTRTransformer
from models.ensemble.Ensemble import Ensemble
from lime.lime_text import LimeTextExplainer
from annotated_text import annotation
from nltk.tokenize import sent_tokenize
import re

pipeline = Pipeline(steps=[('GLTR', GLTRTransformer())])

# instantiate models
svm = SvmTfidfBaseClassifierV2(
    "models/svm_linear_model_90000_features_probability.pkl", "models/tfidf_vectorizer_90000_features.pkl")
bert = BertBaseClassifier("models/model_bertbase_updated.pt")
cnn = CNNGLTRClassifier("models/model_autokeras_gltr_trials_8")
bert_multiclass = BertBaseMultiClassifier("models/model_multiclass.pt")

# instantiate ensemble
models = [bert, cnn, svm]
ensemble = Ensemble(models, ["BERT", "CNN", "SVM"])
weights = np.array([0.25, 0.25, 0.5])


def chunk_into_even_paragraphs(text, max_chunk_size):
    B = len
    I = max_chunk_size
    J = sent_tokenize(text)
    P = max([B(A)for A in J])
    E, C, F = [], [], 0
    for G in J:
        if F+B(G)+B(C) > I:
            E.append(C)
            C, F = [], 0
        C.append(G)
        F += B(G)+B(C)
    E.append(C)
    D = [[B(A)for A in A]for A in E]
    H = True
    while H:
        H = False
        for A in range(B(D)-1, 0, -1):
            K, C = sum(D[A])+B(D[A]), sum(D[A-1])+B(D[A-1])
            L = abs(K-C)
            if (M := K+D[A-1][-1]+1) < I:
                N = C-D[A-1][-1]-1
                O = abs(M-N)
                if O < L:
                    E[A].insert(0, E[A-1].pop())
                    D[A].insert(0, D[A-1].pop())
                    H = True
    return list(map(lambda p: ' '.join(p), E))


def split_into_paragraphs(text, max_char_limit):
    # split the text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)

    output = []
    for paragraph in paragraphs:
        if len(paragraph) > max_char_limit:
            # split the paragraph into sentences
            chunks = chunk_into_even_paragraphs(paragraph, max_char_limit)
            output.extend(chunks)
        else:
            output.append(paragraph)

    return output


def check_if_ai(text: str, threshold: int) -> tuple:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    individual_scores = {}
    for i, text in enumerate(splitted_text):
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        processed_input_df = pipeline.fit_transform(sample_df)
        pred, output_dict = ensemble.predict(
            processed_input_df, weights, threshold)
        scores = {key: val[0] for key, val in output_dict.items()}
        predictions.append(pred)
        individual_scores[i+1] = scores

    # generate the average individual score
    average_scores = {}
    for key, value in individual_scores.items():
        for k, v in value.items():
            if k in average_scores:
                average_scores[k] += v
            else:
                average_scores[k] = v
    for key, value in average_scores.items():
        average_scores[key] = value / len(individual_scores)
    individual_scores["Average"] = average_scores
    return predictions, individual_scores, splitted_text


def check_if_ai_short_text(text: str, threshold: int) -> tuple:
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    processed_input_df = pipeline.fit_transform(sample_df)
    pred, output_dict = ensemble.predict(
        processed_input_df, weights, threshold)
    scores = {key: val[0] for key, val in output_dict.items()}
    return pred, scores


def check_ai_percentage(predictions: list):
    # find the mean of the predictions
    ai_text_count = predictions.count("AI")
    percentage_AI = ai_text_count / len(predictions)
    return percentage_AI


def check_if_paraphrased(text: str) -> bool:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    for text in splitted_text:
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        prediction = bert_multiclass.predict(sample_df)
        predictions.append(prediction[0].detach().numpy())
    return predictions


def check_if_paraphrased_short_text(text: str):
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    prediction = bert_multiclass.predict(sample_df)
    return prediction[0].detach().numpy()


def check_if_paraphrased_percentage(predictions: list, threshold: int):
    # count the number of paraphrased text
    paraphrased_text_count = 0
    for prediction in predictions:
        if prediction[2] > threshold:
            paraphrased_text_count += 1
    percentage_paraphrased = paraphrased_text_count / len(predictions)

    return percentage_paraphrased


def check_if_ai_speed(text: str) -> bool:
    splitted_text = split_into_paragraphs(text, 2000)
    predictions = []
    for text in splitted_text:
        text_lst = [text]
        sample_df = pd.DataFrame(text_lst, columns=['response'])
        prediction = svm.predict(sample_df)
        predictions.append(prediction[0])
    return predictions, splitted_text


def check_if_ai_speed_short_text(text: str) -> bool:
    text_lst = [text]
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    prediction = svm.predict(sample_df)
    return prediction[0]


def check_ai_percentage_speed(predictions: list):
    # find the mean of the predictions
    ai_text_count = predictions.count(1)
    percentage_AI = ai_text_count / len(predictions)
    return percentage_AI


def predict_proba_svm(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    pred = svm.predict_proba(sample_df)
    return np.array(pred)


def predict_proba_bert(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    pred = bert.predict(sample_df)
    returnable = []
    for i in pred:
        temp = i
        returnable.append(np.array([1-temp, temp]))
    return np.array(returnable)


def predict_proba_cnn(text_lst: list):
    sample_df = pd.DataFrame(text_lst, columns=['response'])
    pred = svm.predict(sample_df)
    returnable = []
    for i in pred:
        temp = i
        returnable.append(np.array([1-temp, temp]))
    return np.array(returnable)


def get_explaination(text: str, num_features: int, model: str):
    explainer = LimeTextExplainer(class_names=["Human", "AI"], bow=False)
    if model == "SVM":
        exp = explainer.explain_instance(
            text, predict_proba_svm, num_features=num_features)
    elif model == "BERT":
        exp = explainer.explain_instance(
            text, predict_proba_bert, num_features=num_features)
    elif model == "CNN":
        exp = explainer.explain_instance(
            text, predict_proba_cnn, num_features=num_features)
    return exp.as_html()


def generate_annotated_text(text: list, labels: list, paraphrase_scores=None, paraphrased_threshold=None):
    if paraphrase_scores is None:
        data = []
        colors = {
            "Human": "#afa",
            "AI": "#fea"
        }
        for i in range(len(text)):
            data.append((text[i], labels[i], colors[labels[i]]))
        return data
    else:
        data = []
        colors = {
            "Human": "#afa",
            "AI": "#fea",
        }
        for i in range(len(text)):
            # only create a border around the text if it is paraphrased AI
            if paraphrase_scores[i][2] > paraphrased_threshold and labels[i] == "AI":
                data.append(annotation(
                    text[i], labels[i], colors[labels[i]], border="2px dashed red"))
            else:
                data.append((text[i], labels[i], colors[labels[i]]))
        return data


def generate_annotated_text_speed(text: list, labels: list):
    data = []
    colors = {
        0: "#afa",
        1: "#fea"
    }
    text_labels = {
        0: "Human",
        1: "AI"
    }
    for i in range(len(text)):
        data.append((text[i], text_labels[labels[i]], colors[labels[i]]))
    return data


def has_cyrillic(text: str):
    return bool(re.search('[\u0400-\u04FF]', text))
