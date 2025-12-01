# src/train.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from model import build_model
import timeit



def load_processed_data():
    df = pd.read_parquet(r"H:\ML Intern\zomato\data\preprocessed\data.parquet")

    df.drop([
        'name','rate','votes','listed_in(city)',
        'Rest_SweetOrBakery','Rest_Drink_Oriented_Establishments',
        'Rest_Specialty_Shops','Rest_Dining_Establishments',
        'Rest_Takeaway_and_Delivery','Total_North_Indian',
        'Total_South_Indian','Total_East_Indian','Total_West_Indian',
        'Total_International','Total_Asian','Total_Grill/BBQ/Bar_Food',
        'Total_Fast_Food','Total_Beverages/Desserts',
        'Total_Healthy/Fusion','Total_Bakery'
    ], axis=1, inplace=True)

    x = df.drop("success", axis=1)
    y = df["success"]

    cat_col = ['location','listed_in(type)',"cost_category"]
    num_col = x.select_dtypes(include=['int64','float64']).columns.tolist()

    preprocessor = ColumnTransformer(transformers=[
        ('Encoder', OneHotEncoder(handle_unknown='ignore'), cat_col),
        ('num_scaler', RobustScaler(), num_col)
    ], remainder='drop')

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', build_model())
    ])

    return x, y, pipeline



def train():
    print("=== Training pipeline started ===")

    # Load data + pipeline
    x, y, pipeline = load_processed_data()

    start = timeit.default_timer()

    # ------------------ Cross Validation ------------------
    scores = cross_validate(
        pipeline,
        x,
        y,
        cv=5,
        scoring=[
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro'
        ],
        return_train_score=True
    )

    stop = timeit.default_timer()

    # ------------------ Display Results ------------------
    print("="*50)
    print("Train Accuracy:", scores['train_accuracy'].mean())
    print("Test Accuracy:", scores['test_accuracy'].mean())
    print("-"*50)
    print("Train Precision:", scores['train_precision_macro'].mean())
    print("Test Precision:", scores['test_precision_macro'].mean())
    print("-"*50)
    print("Train Recall:", scores['train_recall_macro'].mean())
    print("Test Recall:", scores['test_recall_macro'].mean())
    print("-"*50)
    print("Train F1:", scores['train_f1_macro'].mean())
    print("Test F1:", scores['test_f1_macro'].mean())
    print("="*50)
    print("Run Time:", stop - start)

    # ------------------ Train full model ------------------
    pipeline.fit(x, y)

    print(f"Saving model to H:\ML Intern\zomato\models\success_model.pkl ...")
    joblib.dump(pipeline, r"H:\ML Intern\zomato\models\success_model.pkl")

    print("=== Training pipeline finished successfully ===")


if __name__ == "__main__":
    train()
