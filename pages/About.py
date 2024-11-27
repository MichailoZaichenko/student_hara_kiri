import streamlit as st

st.title("About Student hara-kiri")
st.markdown(
    """## Overview of Model Building Process
The dataset used for training was the ELI5 HC3 dataset, which can be found https://huggingface.co/datasets/Hello-SimpleAI/HC3. We also added additional data by scraping Student hara-kiri answers and using human-written answers from ELI5 dataset. The model is a weighted average ensemble. It consists of a BERT model, CNN and SVM. 

The models were selected based on two criteria: 
1. Performance on holdout dataset to keep only high-performing models 
2. Diversity to ensure that the models have different decision boundaries for the ensemble to generalise better to unseen data. 

The weighted average in the ensemble was tuned with grid search. On holdout dataset, the ensemble achieve about 96% in accuracy, ROC_AUC score and F1 Score. 

## Overview of Features in GUI
As the ensemble uses several models, it is slightly slower. Thus, for users who are more concerned about speed, the app in "Student hara-kiri Lite" uses only the SVM, which had 95% training accuracy and 95% accuracy on the holdout dataset. 

Our demonstration also provides several additional features for model explainability so that the ensemble is not a black box e.g. showing prediction scores from each models and showing which parts of the text influenced prediction.

We also offer two options when it comes to prediction. The first variation is short text. Under this variation, the text input would be fed to the model as it is. However, if the text is too long, the model might only predict on the first part of the text due to token limits. Thus, for users that want to analyse long text, they can opt for the Essay option, where we would chunk the long text then aggregate the predictions across the different chunks."""
)
