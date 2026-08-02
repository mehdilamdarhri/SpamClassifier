from app.model_loader import load_model


model, vectorizer = load_model()

# Handles prediction logic
def predict_message(message):
    # message.strip() returns True if text exists and False if not. not True becomes False and the opposite
    if not message.strip():
        raise ValueError("Message cannot be empty")
    # vectorizer.transform expect a list and returns 2D list. we provide the outer brackets
    # The vectorizer creates the inner brackets => [[bracket for each messgae]]
    message_vector = vectorizer.transform([message])
    prediction = model.predict(message_vector)[0]

    if prediction == 0:
        return "HAM"

    return "SPAM"