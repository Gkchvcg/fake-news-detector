import joblib
import streamlit as st

# Load model and vectorizer
vectorizer = joblib.load('vectorizer.joblib')
model = joblib.load('model.joblib')

st.title('Fake News Classifier By Maroof Husain')
st.write('This app predicts whether a news is fake or not')

# Input from user
input_text = st.text_input('Enter a news text')

if st.button('Check News'):
    if input_text.strip():
        # Optional: simple cleaning
        cleaned_text = input_text.lower().strip()
        
        input_data = vectorizer.transform([cleaned_text])
        prediction = model.predict(input_data)[0]

        # Check the label mapping
        if prediction == 0:
            st.error('⚠️ This news is FAKE')
        elif prediction == 1:
            st.success('✅ This news is NOT fake')
        else:
            st.warning(f'Unknown prediction: {prediction}')
    else:
        st.write('Bkl kuch likh na')
