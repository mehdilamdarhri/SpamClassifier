import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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


# Turning text into numbers
vectorizer = TfidfVectorizer()

# fit() = learn    transform() = Use what you learned => happens on training data
x_train = vectorizer.fit_transform(x_train)
# transform() on test data
x_test = vectorizer.transform(x_test)


# Use Naive Bayes model as a classifier
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate
predictions = model.predict(x_test)
accuracy = accuracy_score(predictions, y_test)
print(accuracy)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))