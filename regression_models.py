import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR

def run_regression_models(df):
    print("\n🚀 Running Regression Models...")
    df_model = pd.get_dummies(df, columns=['BOROUGH', 'YEAR'], drop_first=True)
    X = df_model.drop(columns=['MEDIAN_PRICE'], errors='ignore')
    y = np.log1p(df_model['MEDIAN_PRICE'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Linear Regression (example)
    lr = LinearRegression().fit(X_train, y_train)
    pred = lr.predict(X_test)
    rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
    print(f"Linear Regression → RMSE: ${rmse:,.0f} | R²: {r2_score(y_test, pred):.3f}")