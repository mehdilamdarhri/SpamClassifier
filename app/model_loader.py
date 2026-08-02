import joblib
import pickle


# Loads saved ML objects
def load_model():
    # Try to execute this code
    try: 
    # The vectorizer is saved to transform future text inputs into the same numerical format the model was trained on.
        model = joblib.load("models/spam_model.pkl")
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

        return model, vectorizer

    # If a file is missing or if file exists but cannot be opened as a valid pickle file, run the code below    
    except FileNotFoundError:
        # Stop the program and report this error
        raise FileNotFoundError("Could not load model files. Train the model first.")
        
    except  pickle.UnpicklingError:
        raise pickle.UnpicklingError("The saved files are corrupted. Train the model again.")