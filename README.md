# SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP).
The project uses **TF-IDF Vectorization** to convert text messages into numerical features and a **Multinomial Naive Bayes classifier** to make predictions.


---
## Project Overview
The goal of this project is to build a machine learning model that can automatically detect whether a message is spam.
The pipeline:
SMS Message
↓
TF-IDF Vectorizer
↓
Multinomial Naive Bayes Model
↓
Spam / Ham Prediction


---
## Dataset
The dataset contains SMS messages labelled as:
- `ham` → normal message
- `spam` → unwanted message

Before training, labels were converted into numbers:
- `0` → ham
- `1` → spam

The data was split into:
- Training data: 80%
- Testing data: 20%

---
## Technologies Used
- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorization
- Multinomial Naive Bayes
- Joblib


---
## Model Training
The model training process:
1. Load and clean the dataset
2. Convert labels into numerical values
3. Split data into training and testing sets
4. Convert text messages into numerical features using TF-IDF
5. Train a Multinomial Naive Bayes classifier
6. Evaluate model performance
7. Save the trained model and vectorizer


---
## Model Evaluation
The model was evaluated using:

### Accuracy
Measures how many predictions were correct overall.

### Precision
Measures how often the model is correct when it predicts spam.

### Recall
Measures how many actual spam messages the model successfully detects.

### F1-score
Balances precision and recall.


---
## Results
Model performance:
Accuracy: 90%

Precision: 100%

Recall: 75%

F1-score: 86%

## Making Predictions
The prediction script allows users to enter a new message and classify it.

Example:

Input: Congratulations! You won a free prize!
Output: SPAM

Input: Are we still meeting today?
Output: HAM


## Project Structure
SpamClassifier/

├── data/
│ └── spam.csv
│
├── models/
│ ├── spam_model.pkl
│ └── tfidf_vectorizer.pkl
│
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore



---
## How to Run
* Clone the repository:
git clone <repository-url>

* Install dependencies:
pip install -r requirements.txt

* Train the model:
python train_model.py

* Run prediction:
python predict.py