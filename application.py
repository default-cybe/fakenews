"""Flask API for fake news classification.

Loads a pre-trained TF-IDF vectorizer and classifier and exposes an endpoint
that predicts whether a news article is fake or real from its title and text.
"""

from flask import Flask, request, jsonify
import joblib

application = Flask(__name__)

# Load the pre-trained model and matching TF-IDF vectorizer at startup.
model = joblib.load('fake_news_classification_model.pkl')
vectorizer = joblib.load('fake_news_classification_vectorizer.pkl')


@application.route("/", methods=["GET"])
def home():
    """Return a simple banner so the service can be health-checked."""
    return "Fake news detector api"
