# Student hara-kiriStudent hara-kiri: an advanced ChatGPT detector

Student hara-kiri is a project created by 3 students from DNU. It features an ensemble with an accuracy rate of 96%. For more information, visit us on our github page https://github.com/MichailoZaichenko/student_hara_kiri

## Features

Student hara-kiri has 5 main features:

1. ChatGPT content detection
2. Paraphrased ChatGPT content detection
3. Probability breakdown by each model (to know each model’s prediction)
4. Explainability generation
5. Cyrillic character detection (to prevent bypassing by mapping English alphabets to Cyrillic characters)

### ChatGPT Content Detection

Student hara-kiri provides a way for users to validate if a piece of text is written by humans or ChatGPT. The web app has 2 variations, “Essay” and “Short-Text”.

<p align="center">
  <img src="./images/gptgone_variations.jpg" width="75%" alt="GPTGone Variations"/>
</p>

When using the “Essay” variation, there will be no character limit on the size of input a person can enter as the input will split into smaller chunks when the models’ token limit is exceeded. On the other hand, when using the “Short-Text” variation, the user input will be truncated when it exceeds the model’s token limit, however, it will return a more accurate prediction than the Essay variation since our model was trained on the whole text. Hence it is most suited for text that is less than 2000 characters long, given the token limit for BERT is 512 tokens.

<p align="center">
  <img src="./images/gptgone_sample_output.jpg" width="85%" alt="GPTGone Sample Output"/>
</p>

### Paraphrased ChatGPT Content Detection

Student hara-kiri is also capable of detecting ChatGPT-generated content that was paraphrased using a paraphraser like T5. Our model is currently only trained with T5 paraphrased inputs, in the future, we may explore other paraphrasers to bolster the robustness of the model.

<p align="center">
  <img src="./images/gptgone_sample_paraphrased_output.jpg" width="85%" alt="GPTGone Sample Paraphrased Output"/>
</p>

### Probability Breakdown by Each Model

GPTGone is able to display the individual predictions of each model, for users who are curious about the predictions of the constituent models in the ensemble.

<p align="center">
  <img src="./images/gptgone_sample_probability_breakdown.jpg" width="85%" alt="GPTGone Sample Probability Breakdown"/>
</p>

### Explainability Generation

For the users that would like to understand why their input is classified as AI-written or human-written, GPTGone offers an explanation generator as well. Currently, the explanation generator only supports explanations by our SVM model due to the expensive computational costs of the BERT and CNN models.

<p align="center">
  <img src="./images/gptgone_sample_explanation.jpg" width="85%" alt="GPTGone Sample Explanation"/>
</p>

### Cyrillic Character Detection

Another important feature of GPTGone is the capability of detecting the Cyrillic alphabet, to
prevent bypassers from converting text to the Cyrillic alphabet and bypassing GPTGone.

<p align="center">
  <img src="./images/gptgone_cyrillic_character_detection.jpg" width="85%" alt="GPTGone Cyrillic Character Detection"/>
</p>

## How to run Student hara-kiri locally

Requirement: python version >= 3.9

1. If git lfs is not installed, install it using this command:

   ```bash
   git lfs install
   ```
2. Clone the repository from https://github.com/MichailoZaichenko/student_hara_kiri

   ```bash
   git clone https://github.com/MichailoZaichenko/student_hara_kiri
   ```
3. Change your directory into Student hara-kiriGPTGon using:

   ```bash
   cd Student hara-kiri
   ```
4. Create a new virtual environment at the root of the project directory.

   ```bash
   py -m venv venv
   ```
5. Activate the virtual environment.

   Mac:

   ```bash
   source ./venv/bin/activate
   ```
   Windows:

   ```bash
   venv\Scripts\Activate.ps1
   ```
6. Install the requirements using:

   ```bash
   pip install -r requirements.txt
   ```
7. Run this command to spin up the local server:

   ```
   streamlit run App.py
   ```
8. Visit the URL displayed on the terminal to open up the web application.
