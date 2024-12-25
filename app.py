# Write the Streamlit code into app.py
with open('app.py', 'w') as f:
    f.write(
import streamlit as st
import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download NLTK Data
nltk.download('stopwords')

# Load Model and Vectorizer
model = pickle.load(open('best_model.pkl', 'rb'))
vectorizer = pickle.load(open('count_vectorizer.pkl', 'rb'))

# Preprocessing Function
sw = set(stopwords.words('english'))
def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

# Classification Function
def text_classification(text):
    cleaned_review = text_preprocessing(text)
    process = vectorizer.transform([cleaned_review]).toarray()
    prediction = model.predict(process)
    return prediction[0]

# Streamlit App
def main():
    st.title("Fake Review Detection Model")
    st.write("Enter a review to check if it's legitimate or fraudulent.")
    
    review = st.text_area("Enter Review:")
    if st.button("Classify"):
        if review:
            with st.spinner("Processing..."):
                result = text_classification(review)
                if result:
                    st.success("The review is **Legitimate**.")
                else:
                    st.error("The review is **Fraudulent**.")
        else:
            st.warning("Please enter a review to classify.")

if __name__ == '__main__':
    main()
    )
