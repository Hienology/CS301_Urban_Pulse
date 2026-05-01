import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def run_classification_models(df):
    print("\n🚀 Running Classification Models...")
    df_model = pd.get_dummies(df, columns=['BOROUGH', 'YEAR'], drop_first=True)
    y_class = (df_model['MEDIAN_PRICE'] > df_model['MEDIAN_PRICE'].median()).astype(int)
    X = df_model.drop(columns=['MEDIAN_PRICE'], errors='ignore')
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25, random_state=42)

    logreg = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    print(f"Logistic Regression Accuracy: {accuracy_score(y_test, logreg.predict(X_test)):.3f}")