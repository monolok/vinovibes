import streamlit as st
from joblib import load
import pandas as pd
import json

# Load your model and feature encoder
regressor = load('vinovibes.joblib')
feature_encoder = load('feature_encoder_vinovibes.joblib')
# Function to load words from a JSON file
def load_words_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
# Load words in your Streamlit app
word_list = load_words_from_json('words.json')
province = load_words_from_json('provinces.json')
variety = load_words_from_json('varieties.json')

# Streamlit app setup
st.title('Wine Price and Variety Predictor')

# User inputs for numerical and categorical features
province = st.selectbox('Province', province)
variety = st.selectbox('Variety', variety)
points = st.slider('Points', 80, 100, 90)
sweet = st.slider('Sweetness', min_value=0.0, max_value=10.0, value=1.4)
acidity = st.slider('Acidity', min_value=0.0, max_value=10.0, value=2.9)
body = st.slider('Body', min_value=0.0, max_value=10.0, value=2.05)
tannin = st.slider('Tannin', min_value=0.0, max_value=10.0, value=2.82)
abv = st.slider('ABV', min_value=0.0, max_value=20.0, value=13.02)

# Use word_list for the dropdown
selected_word = st.selectbox('Select a Word', word_list)


# Predict button
if st.button('Predict'):
    # Prepare input data with all features
    df = pd.DataFrame([{
        'points': points,
        'province': province,
        'variety': variety,
        'sweet': sweet,
        'acidity': acidity,
        'body': body,
        'tannin': tannin,
        'abv': abv,
        selected_word: 1  # Mark the selected word as present
    }])

    # Fill other word columns with 0
    for word in word_list:
        if word not in df.columns:
            df[word] = 0

    # Transform features using the feature_encoder
    encoded_input = feature_encoder.transform(df)

    # Make prediction
    prediction = regressor.predict(encoded_input)
    
    # Show prediction
    st.write(f'Predicted Price: {prediction[0]}')

# Run this with `streamlit run app.py`
