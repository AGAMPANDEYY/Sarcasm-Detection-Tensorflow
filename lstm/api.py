from flask import Flask,request,jsonify
import torch
import torch
import tensorflow as tf
import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app= Flask(__name__)
@app.route("/predict",methods=['POST'])

max_features = 2500
tokenizer = Tokenizer(num_words=max_features, split=' ')
#tokenizer.fit_on_texts(df['headline'].values)


def predict():
  if model:
    try:
      def preprocess_text(text):
        text = text.replace('rt', ' ')
        text = [text]
        sequences = tokenizer.texts_to_sequences(text)
        sequences = pad_sequences(sequences, maxlen=25) # Assuming max length of sequences is 25
        return sequences
      text=request.data.decode('utf-8')
      preprocessed_text = preprocess_text(text)
      prediction=model.predict(preprocessed_text)
  
  else:
        print ('Train the model first')
        return ('No model here to use')

if __name__=="__main__":

  model=torch.load("lstm/model_lstm_e1.pt")
  print("Model loaded")
  app.run(port=12345,debug=True)
