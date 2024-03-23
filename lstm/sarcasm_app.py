
import streamlit as st
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from keras.layers import InputLayer, Embedding, SpatialDropout1D, LSTM
from keras.models import Sequential
import json



loaded_model=torch.load("lstm/model_lstm_e1.pt")

# Load data
df1 = pd.read_json('Sarcasm-Detection-Tensorflow/Dataset/Sarcasm_Headlines_Dataset.json', lines=True)
df2 = pd.read_json('Sarcasm-Detection-Tensorflow/Dataset/Sarcasm_Headlines_Dataset_v2.json', lines=True)
frames = [df1, df2]
df = pd.concat(frames)

# Tokenize to vectorize and convert texts into features
max_features = 2500
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(df['headline'].values)

# Function to preprocess input text
def preprocess_text(text):
    text = text.replace('rt', ' ')
    text = [text]
    sequences = tokenizer.texts_to_sequences(text)
    sequences = pad_sequences(sequences, maxlen=25) # Assuming max length of sequences is 25
    return sequences

# Function to predict using the loaded Keras model
def predict_sarcasm(text):
    preprocessed_text = preprocess_text(text)
    prediction = loaded_model.predict(preprocessed_text)
    return prediction

# Streamlit app
st.title("Sarcasm Detection App")

user_input = st.text_input("Enter your text:")

if st.button("Predict"):
    prediction = predict_sarcasm(user_input)
    sarcasm_probability = prediction[0][1]
    st.write(f"Sarcasm Probability: {sarcasm_probability:.2f}")
    if sarcasm_probability > 0.5:
        st.write("This text is sarcastic.")
    else:
        st.write("This text is not sarcastic.")
