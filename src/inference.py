import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys

MODEL_PATH = r"H:\ML Intern\zomato\models\success_model.pkl"

def load_trained_model(model_path: str = MODEL_PATH):
    """
    Load the trained model from disk.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            f"Did you run the training pipeline first?"
        )

    pipeline = joblib.load(model_path)
    return pipeline


# -------------------------
# Load model ONCE globally
# -------------------------
pipeline = load_trained_model()


def predict(online_order, book_table, phone, location, approx_cost, menu_item, type_rest, cost_category):

    test_df = pd.DataFrame([{
        'online_order': online_order,
        'book_table': book_table,
        'phone': phone,
        'location': location,
        'approx_cost(for two people)': approx_cost,
        'menu_item': menu_item,
        'listed_in(type)': type_rest,
        'cost_category': cost_category
    }])

    predicted = pipeline.predict(test_df)[0]
    predicted = "Success" if predicted == 1 else "Fail"
    return predicted
