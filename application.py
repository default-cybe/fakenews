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


@application.route('/predict', methods=['POST'])
def predict():
    """Classify an article from its title and text.

    Expects a JSON body with ``title`` and ``text`` fields. The two are
    concatenated, transformed by the vectorizer, and scored by the model.
    Returns the predicted label as a string, or a 400 error on bad input.
    """
    try:
        data = request.json

        title = data['title']
        text = data['text']

        # Combine title and body, then convert to the TF-IDF feature vector.
        input_text = title + ' ' + text
        input_vector = vectorizer.transform([input_text])

        prediction = model.predict(input_vector)[0]
        prediction_str = str(prediction)

        response = {
            'prediction': prediction_str
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    application.run(debug=True)
