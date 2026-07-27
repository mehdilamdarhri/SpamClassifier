import joblib


# The vectorizer is saved to transform future text inputs into the same numerical format the model was trained on.
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
model = joblib.load("models/spam_model.pkl")

message = input("Enter a message: ")
# vectorizer.transform expect a list and returns 2D list. we provide the outer brackets
# The vectorizer creates the inner brackets => [[bracket for each messgae]]
message_vector = vectorizer.transform([message])

prediction = model.predict(message_vector)[0]

if prediction == 0:
    print("HAM")
else:
    print("SPAM")