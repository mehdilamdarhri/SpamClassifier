import joblib

# Loads saved ML objects
def load_model():
    # The vectorizer is saved to transform future text inputs into the same numerical format the model was trained on.
    model = joblib.load("models/spam_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    return model, vectorizer