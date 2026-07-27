import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Convert file to csv
#df = pd.read_csv("data/SMSSpamCollection", sep="\t", names=["label", "message"])
#df.to_csv("data/spam.csv", index = False)

df = pd.read_csv("data/spam.csv")
#print(df.info())
#print(df.isnull().sum())
#print(df.head())
#print(df.shape)



# Check the classes
#print(df["label"].value_counts())

# Convert ham spam into 0 and 1
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})



# Splitting Data
x = df["message"]
y = df["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# Convert text messages into numerical features
vectorizer = TfidfVectorizer()

# fit() = learn    transform() = Use what you learned => happens on training data
x_train = vectorizer.fit_transform(x_train)
# transform() on test data
x_test = vectorizer.transform(x_test)
# Save the vectorizer
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


# Use Naive Bayes model as a classifier
model = MultinomialNB()
model.fit(x_train, y_train)
# Save the model
joblib.dump(model, "models/spam_model.pkl")

# Evaluate
predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print(accuracy)

# precision: when the model says "spam" or "ham" how often is it right ?
# recall: of all the actual messages. how many did it catch ?
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))