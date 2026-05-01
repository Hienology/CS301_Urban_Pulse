import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

def run_regression_models(df):
    """
    Runs multiple supervised regression models and saves results + plots.
    """
    print("🚀 Running Supervised Regression Models...")
    print("Target: log(MEDIAN_PRICE) | Features: Crime types + Borough + Year\n")

    # Create output directory
    os.makedirs("output", exist_ok=True)

    # Prepare data
    df_model = pd.get_dummies(df, columns=['BOROUGH', 'YEAR'], drop_first=True)
    X = df_model.drop(columns=['MEDIAN_PRICE'], errors='ignore')
    y = np.log1p(df_model['MEDIAN_PRICE'])   # log transform for better modeling

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    results = []

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    pred_lr = lr.predict(X_test)
    rmse_lr = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_lr)))
    r2_lr = r2_score(y_test, pred_lr)
    results.append(['Linear Regression', rmse_lr, r2_lr])

    # 2. Polynomial Regression (degrees 1-4)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    for degree in range(1, 5):
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train_scaled)
        X_test_poly = poly.transform(X_test_scaled)

        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        pred = model.predict(X_test_poly)
        rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred)))
        r2 = r2_score(y_test, pred)
        results.append([f'Polynomial Degree {degree}', rmse, r2])

    # 3. Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    rmse_rf = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_rf)))
    r2_rf = r2_score(y_test, pred_rf)
    results.append(['Random Forest', rmse_rf, r2_rf])

    # 4. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42)
    gb.fit(X_train, y_train)
    pred_gb = gb.predict(X_test)
    rmse_gb = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_gb)))
    r2_gb = r2_score(y_test, pred_gb)
    results.append(['Gradient Boosting', rmse_gb, r2_gb])

    # 5. SVR (RBF)
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    pred_svr = svr.predict(X_test_scaled)
    rmse_svr = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_svr)))
    r2_svr = r2_score(y_test, pred_svr)
    results.append(['SVR (RBF)', rmse_svr, r2_svr])

    # 6. Decision Tree
    dt = DecisionTreeRegressor(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    pred_dt = dt.predict(X_test)
    rmse_dt = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(pred_dt)))
    r2_dt = r2_score(y_test, pred_dt)
    results.append(['Decision Tree', rmse_dt, r2_dt])

    # =============================================================================
    # RESULTS TABLE
    # =============================================================================
    results_df = pd.DataFrame(results, columns=['Model', 'RMSE ($)', 'R²'])
    results_df = results_df.sort_values('RMSE ($)')
    
    print("\n📊 SUPERVISED REGRESSION MODEL COMPARISON")
    print(results_df.round(3))

    # Top Feature Importance from Random Forest
    print("\n🔍 Top 10 Feature Importance (Random Forest)")
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10))

    # Save comparison plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x='RMSE ($)', y='Model', palette='viridis')
    plt.title('Regression Model Comparison - RMSE (Lower is better)')
    plt.xlabel('RMSE ($)')
    plt.tight_layout()
    plt.savefig('output/regression_model_comparison.png')
    plt.close()

    print("\n✅ Regression models completed!")
    print("   → Results table printed above")
    print("   → Feature importance shown")
    print("   → Plot saved: output/regression_model_comparison.png")

    return results_df, importances
