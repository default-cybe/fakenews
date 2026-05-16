# Fake News Detector API

A lightweight Flask API that classifies news articles as fake or real using a
pre-trained scikit-learn model. Send a headline and body text to the API and it
returns the model's prediction.

## How it works

The service loads two artifacts at startup with `joblib`:

- `fake_news_classification_vectorizer.pkl`: a TF-IDF vectorizer that turns the
  raw article text into a sparse feature vector.
- `fake_news_classification_model.pkl`: the trained classifier that maps the
  TF-IDF vector to a label.

On each request the title and text are concatenated into a single string,
transformed by the vectorizer, and passed to the model's `predict` method. The
resulting label is returned as a string.

## API endpoints

### `GET /`

Health/landing endpoint. Returns a plain-text banner:

```
Fake news detector api
```

### `POST /predict`

Classifies an article. Expects a JSON body with `title` and `text` fields.

Request body:

```json
{
  "title": "Article headline",
  "text": "Full body of the article..."
}
```

Success response (`200`):

```json
{
  "prediction": "1"
}
```

On error (for example, missing fields or malformed input) it returns `400` with:

```json
{
  "error": "description of what went wrong"
}
```

