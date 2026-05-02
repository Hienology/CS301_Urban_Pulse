"""
main.py
Orchestrates the entire Urban Pulse NYC analysis pipeline.
"""

import os
from data_loading import load_and_aggregate_data
from eda import run_eda
from hypothesis_testing import run_hypothesis_test
from regression_models import run_regression_models
from classification_models import run_classification_models
from tree_ensemble_models import run_tree_ensemble_models
from clustering import run_hierarchical_clustering
from within_neighborhood_analysis import run_within_neighborhood_analysis

print("="*80)
print("URBAN PULSE NYC - FULL ANALYSIS PIPELINE")
print("="*80)

# Create output directory
os.makedirs("output", exist_ok=True)

# Step 1: Load and aggregate data
df, housing = load_and_aggregate_data()

# Step 2: EDA
run_eda(df)

# Step 3: Hypothesis Testing
run_hypothesis_test(df)

# Step 4: Regression Models
run_regression_models(df)

# Step 5: Classification Models
run_classification_models(df)

# Step 6: Tree & Ensemble Models
run_tree_ensemble_models(df)

# Step 7: Hierarchical Clustering + Dendrogram
run_hierarchical_clustering(df)

# Step 8: Within-Neighborhood Analysis
run_within_neighborhood_analysis(df, housing)

print("\n🎉 FULL ANALYSIS PIPELINE COMPLETED!")
print("Visual results are in 'output/' folder.")
