from flask import Flask, render_template, request, redirect, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle
import nltk
import string
from nltk.corpus import stopwords
import pandas as pd
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier

# Initialize the Flask app
app = Flask(__name__)

nltk.download('stopwords')

# Load the model
with open('RF_model.pkl', 'rb') as model_file:
    rf = pickle.load(model_file)

df=pd.read_csv("C:/Users/Malavika/Desktop/Depression/preprocessedsec.csv")

# Load the processed messages for TF-IDF vectorization
processed_messages = df['processed_messages'].astype(str).tolist() # You need to load your dataset here

# Initialize TF-IDF vectorizer and fit on the dataset
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=0.75, min_df=5, max_features=10000)
tfidf = tfidf_vectorizer.fit_transform(processed_messages)

# Preprocess the input text
def preprocess_input_text(sample):
    stemmer = PorterStemmer()

    # Define the list of extended stopwords
    extended_stopwords = nltk.corpus.stopwords.words("english")
    other_exclusions = ["#ff", "ff", "rt"]
    extended_stopwords.extend(other_exclusions)

    # Preprocessing steps
    tweet_space = re.sub(r'\s+', ' ', sample)
    tweet_name = re.sub(r'@[\w\-]+', '', tweet_space)
    tweet_no_links = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', tweet_name)
    tweet_no_punctuation = re.sub("[^a-zA-Z]", " ", tweet_no_links)
    tweet_stripped = tweet_no_punctuation.strip()
    tweet_no_numbers = re.sub(r'\d+(\.\d+)?', 'numbr', tweet_stripped)
    tweet_lower = tweet_no_numbers.lower()
    tokenized_tweet = tweet_lower.split()
    tokenized_tweet = [stemmer.stem(token) for token in tokenized_tweet if token not in extended_stopwords]
    processed_input_text = ' '.join(tokenized_tweet)

    return processed_input_text

# Route to the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to the form where user inputs text
@app.route('/input')
def input_form():
    return render_template('input_form.html')

# Route to process the form input and display the result
@app.route('/result', methods=['POST'])
def result():
        
        # Get the input text from the form
        sample = request.form['input_text']
        
        # Preprocess the input text
        processed_input_text = preprocess_input_text(sample)
        
        # Transform the processed text using the trained TF-IDF vectorizer
        tfidf_sample = tfidf_vectorizer.transform([processed_input_text])
        
        # Make predictions
        predicted_label = rf.predict(tfidf_sample)[0]
        
        # Map class labels to their meanings
        class_labels = {
            0: "Non Depressive",
            1: "Depressive"
        }
        
        predicted_label_text = class_labels[predicted_label]

        # Pass the prediction and input text to the result.html
        return render_template('result.html', input_text=sample, predicted_label=predicted_label_text)

if __name__ == '__main__':
    app.run(debug=True)
