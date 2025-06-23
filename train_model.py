import warnings
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

warnings.filterwarnings("ignore")

def train_and_save_model():
    """Trains the Iris classification model and saves it."""
    print("Loading Iris dataset...")
    d = load_iris()
    df = pd.DataFrame(d.data, columns=d.feature_names)
    df["target"] = d.target

    x = df.drop("target", axis="columns")
    y = df["target"]

    print("Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

    print("Training Logistic Regression model...")
    # Increased max_iter for convergence, and added solver for robustness
    model = LogisticRegression(max_iter=1000, solver='lbfgs')
    model.fit(x_train, y_train)

    model_path = os.path.join(os.path.dirname(__file__), 'iris.pkl')
    print(f"Saving trained model to {model_path}...")
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

if __name__ == "__main__":
    train_and_save_model()