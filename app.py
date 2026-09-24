from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
import requests
from textblob import TextBlob

# Initialize Flask app
app = Flask(__name__)

# Directory containing CSV files
CSV_DIR = 'csv_files/'

# Initialize app data storage
app_data = {}

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def correct_spelling(text):
    blob = TextBlob(text)
    return str(blob.correct())

# Preprocessing function
def preprocess(text, use_stopwords=True):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric characters
    tokens = nltk.word_tokenize(text.lower())
    if use_stopwords:
        tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stemmed_tokens = [stemmer.stem(token) for token in lemmatized_tokens]
    return ' '.join(stemmed_tokens)

# Function to fetch data from the API
def fetch_data_from_api(app_name):
    api_url = f"http://127.0.0.1:5000/{app_name.lower()}"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        questions = [item['question'] for item in data]
        answers = [item['answer'] for item in data]
        return questions, answers
    else:
        print(f"Error: Unable to fetch data from API. Status code: {response.status_code}")
        return None, None

# Function to create TF-IDF vectorizer and fit on the data
def create_vectorizer(questions_list):
    vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize)
    X = vectorizer.fit_transform([preprocess(q) for q in questions_list])
    return vectorizer, X

def get_response(text, questions_list, answers_list, vectorizer, X):
    text = correct_spelling(text)
    processed_text = preprocess(text)
    vectorized_text = vectorizer.transform([processed_text])
    similarities = cosine_similarity(vectorized_text, X).flatten()

    # Thresholds
    close_match_threshold = 0.9
    unique_question_threshold = 0.95
    top_n = 3

    # Sort indices by similarity in descending order
    sorted_indices = similarities.argsort()[::-1]

    # Check for a close match
    if similarities[sorted_indices[0]] >= close_match_threshold:
        return [{
            "question": questions_list[sorted_indices[0]],
            "answer": answers_list[sorted_indices[0]],
        }]

    # Collect top N distinct matches
    top_matches = []
    added_indices = set()  # Track added indices to avoid duplicates

    for idx in sorted_indices:
        if similarities[idx] > 0:
            is_unique = all(
                cosine_similarity(X[idx], X[added_idx])[0][0] < unique_question_threshold
                for added_idx in added_indices
            )
            if is_unique:
                top_matches.append({
                    "question": questions_list[idx],
                    "answer": answers_list[idx],
                })
                added_indices.add(idx)

        if len(top_matches) == top_n:
            break

    return top_matches if top_matches else "I can't answer this question."


# Load CSV files into app_data
def load_data():
    for csv_file in os.listdir(CSV_DIR):
        if csv_file.endswith('.csv'):
            app_name = os.path.splitext(csv_file)[0]
            df = pd.read_csv(os.path.join(CSV_DIR, csv_file))
            app_data[app_name] = df

# Create routes dynamically for each app's FAQ
def create_routes():
    for app_name in app_data:
        def route_function(app_name=app_name):
            def inner_function():
                return jsonify(app_data[app_name].to_dict(orient='records'))
            return inner_function

        app.add_url_rule(f'/{app_name}', app_name, route_function(app_name))

# Load data on app startup
load_data()

# Create API routes
create_routes()

@app.route('/', methods=['GET', 'POST'])
def index():
    response = None
    selected_app = None  # Track selected app
    if request.method == 'POST':
        app_name = request.form['app_name']
        selected_app = app_name 
        question = request.form['question']
        questions_list, answers_list = fetch_data_from_api(app_name)
        if questions_list and answers_list:
            vectorizer, X = create_vectorizer(questions_list)
            response = get_response(question, questions_list, answers_list, vectorizer, X)
        else:
            response = [{"question": "Error", "answer": "Unable to fetch data."}]

    return render_template('index.html', response=response, selected_app=selected_app)

if __name__ == '__main__':
    app.run(debug=True)
