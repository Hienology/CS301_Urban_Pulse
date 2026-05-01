import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def run_tree_ensemble_models(df):
    print("\n🚀 Running Tree & Ensemble Models...")
    df_model = pd.get_dummies(df, columns=['BOROUGH', 'YEAR'], drop_first=True)
    X = df_model.drop(columns=['MEDIAN_PRICE'], errors='ignore')
    y = np.log1p(df_model['MEDIAN_PRICE'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Pruned Decision Tree
    dt = DecisionTreeRegressor(max_depth=4, min_samples_split=5, random_state=42)
    dt.fit(X_train, y_train)
    plt.figure(figsize=(16, 8))
    plot_tree(dt, feature_names=X.columns.tolist(), filled=True, rounded=True, fontsize=10)
    plt.title("Pruned Decision Tree")
    plt.savefig("output/pruned_decision_tree.png")
    plt.close()

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, random_state=42).fit(X_train, y_train)
    print("Gradient Boosting ready (feature importance available)")