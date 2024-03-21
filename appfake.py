import streamlit as st
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
import string
from tensorflow.keras.models import load_model
# Download NLTK data (only needed once)
nltk.download('punkt')

# Load the pre-trained model
model = load_model("best_model.h5")

# Text preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w', '', text)
    return text

st.title("Fake News Detector")

# User input
news_input = st.text_area("Enter the news text here:")

if st.button("Predict"):
    if news_input:
        # Tokenize and preprocess the input data
        new_input_data = [news_input]
        new_sequences = [wordopt(sentence) for sentence in new_input_data]

        tokenizer1 = Tokenizer()
        tokenizer1.fit_on_texts(new_sequences)
        new_sequences = tokenizer1.texts_to_sequences(new_sequences)

        new_sequences = pad_sequences(new_sequences, maxlen=8280)

        new_data = np.array(new_sequences)

        new_data = pad_sequences(new_data, maxlen=8280, padding='post', truncating='post')

        # Make predictions
        predictions = model.predict(new_data)
        predicted_labels = [1 if score >= 0.5 else 0 for score in predictions]
        class_mapping = {0: "fake", 1: "true"}
        predicted_class_names = [class_mapping[label] for label in predicted_labels]

        st.write("Prediction:", predicted_class_names[0])

    else:
        st.warning("Please enter news text to predict.")

