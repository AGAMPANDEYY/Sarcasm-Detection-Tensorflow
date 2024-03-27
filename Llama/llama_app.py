# -*- coding: utf-8 -*-
"""llama_app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1RWSzQso-1qDLSd5MFkS0wgcg2xx1vdic
"""

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import transformers
from peft import *
import os


#access_token="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"

#os.environ["HUGGINGFACE_TOKEN"] = access_token

st.set_page_config(
    page_title="Sarcasm Detection App",
    page_icon="😏",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.image("dsg_logo.pn")
st.sidebar.title("Data Science Group,IITR Sarcasm Detection Project")
# Customize the theme
custom_theme = """
    [theme]
    primaryColor = "#E694FF"
    backgroundColor = "#00172B"
    secondaryBackgroundColor = "#0083B8"
    textColor = "#DCDCDC"
    font = "sans serif"
"""
# Apply the custom theme
st.markdown(f'<style>{custom_theme}</style>', unsafe_allow_html=True)

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#loading the model

model_id="AgamP/llama-sarcasm"

model=AutoModelForCausalLM.from_pretrained(
    model_id,
    ).to(device)

tokenizer=AutoTokenizer.from_pretrained(model_id)

generation_config = model.generation_config
generation_config.max_new_tokens = 100
generation_config_temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config_eod_token_id = tokenizer.eos_token_id

def predict(input_text,max_length):
   prompt = f"""
   <human>: {input_text} Is the sentence sarcastic?
   <assistant>:
   """.strip()

   encoding = tokenizer(prompt, return_tensors="pt").to(device)
   with torch.inference_mode():
    outputs = model.generate(
      input_ids=encoding.attention_mask,
      generation_config=generation_config,
    )
   return tokenizer.decode(outputs[0],skip_special_tokens=True)



st.title("Sarcasm Detection App")

input_text = st.text_input("Enter your text here:", placeholder="Type your text here...")

predict_button = st.button("Predict")

if predict_button:
        prediction = predict(input_text, max_length=100)
        st.write("Prediction:", prediction)

        #if prediction == 1:
            #st.write("Prediction: 😏 Sarcastic!")
        #else:
            #st.write("Prediction: 😊 Not Sarcastic")

# Frontend improvements
st.write("""
### About the Models
- **LSTM Model:** This model utilizes a Long Short-Term Memory (LSTM) neural network architecture trained on a dataset of sarcastic and non-sarcastic headlines. It preprocesses the text using a tokenizer and then makes predictions based on the learned patterns.
- **BERT Model:** This model employs a pre-trained BERT (Bidirectional Encoder Representations from Transformers) model fine-tuned on a sarcasm detection task. BERT is a powerful transformer-based model that captures contextual information from the input text, allowing it to make accurate predictions.
- **Note:** While these models can detect sarcasm with high accuracy, they may not always provide the correct answer due to the inherent complexity and ambiguity of sarcasm.
- **About the Dataset:** This model was trained on a dataset consisting of headlines. It has learned patterns from various headlines to distinguish between sarcastic and non-sarcastic statements. While it performs well on headlines, its accuracy may vary depending on the context and complexity of the text.
""")
