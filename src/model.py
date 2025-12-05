# src/model.py
from xgboost import XGBClassifier
from data_pipeline import preprosses
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def build_model_pipeline():
    pipeline = Pipeline([
        ('preprocessor', preprosses()),
        ('model', XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,

        # Best Params
        colsample_bytree=1,
        gamma=0.2,
        learning_rate=0.2,
        max_depth=7,
        n_estimators=200,
        subsample=0.8
            ))
    ])
    return pipeline