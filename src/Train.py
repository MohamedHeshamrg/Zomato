# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from model import build_model_pipeline
import timeit
from data_pipeline import preprosses
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error



def load_data():
    X_train = pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\X_train.parquet")
    y_train =  pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\Y_train.parquet")

    X_test =  pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\X_test.parquet")
    y_test =  pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\Y_test.parquet")

    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, y_train, X_test, y_test):
    scores = {}

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Accuracy
    scores['train_accuracy'] = accuracy_score(y_train, y_train_pred)
    scores['test_accuracy'] = accuracy_score(y_test, y_test_pred)

    # Precision Macro
    scores['train_precision_macro'] = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    scores['test_precision_macro'] = precision_score(y_test, y_test_pred, average='macro', zero_division=0)

    # Recall Macro
    scores['train_recall_macro'] = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    scores['test_recall_macro'] = recall_score(y_test, y_test_pred, average='macro', zero_division=0)

    # F1 Macro
    scores['train_f1_macro'] = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    scores['test_f1_macro'] = f1_score(y_test, y_test_pred, average='macro', zero_division=0)

    # Print scores
    print("="*50)
    print("Train Accuracy:", scores['train_accuracy'])
    print("Test Accuracy:", scores['test_accuracy'])
    print("-"*50)
    print("Train Precision:", scores['train_precision_macro'])
    print("Test Precision:", scores['test_precision_macro'])
    print("-"*50)
    print("Train Recall:", scores['train_recall_macro'])
    print("Test Recall:", scores['test_recall_macro'])
    print("-"*50)
    print("Train F1:", scores['train_f1_macro'])
    print("Test F1:", scores['test_f1_macro'])
    print("="*50)



    return scores



    print("=== Training pipeline finished successfully ===")
def train():
    print("=== Training pipeline started ===")
    start = timeit.default_timer()

    # ------------------ Load Data ------------------
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)},Test size: {len(X_test)}")

    # ------------------ Build Model ------------------
    print("Building model pipeline...")
    model = build_model_pipeline()

    # ------------------ Train ------------------
    print("Training model...")
    model.fit(X_train, y_train)
    print("Training finished.")

    # ------------------ Evaluate ------------------
    print("\n=== Evaluating model performance ===")
    scores = evaluate_model(model, X_train, y_train, X_test, y_test)

    # ------------------ Save Model ------------------
    model_path = r"H:\ML Intern\zomato\models\success_model.joblib"
    print(f"\nSaving model to {model_path} ...")
    joblib.dump(model, model_path)

    stop = timeit.default_timer()
    print(f"\n=== Training pipeline finished successfully in {stop - start:.2f} seconds ===")

if __name__ == "__main__":
    train()

