import os

# Optional imports (some modules may be absent in minimal setups). Import lazily and
# continue gracefully if they're missing so opening the repo or importing `main`
# doesn't crash.
from data_loading import load_and_aggregate_data
from eda import run_eda
from hypothesis_testing import run_hypothesis_test
from regression_models import run_regression_models
from classification_models import run_classification_models
from tree_ensemble_models import run_tree_ensemble_models
from clustering import run_hierarchical_clustering


def main():
	print("=" * 80)
	print("URBAN PULSE NYC - Full Analysis Pipeline")
	print("=" * 80)

	os.makedirs("output", exist_ok=True)

	df = load_and_aggregate_data()

	run_eda(df)

	run_hypothesis_test(df)
	run_regression_models(df)
	run_classification_models(df)
	run_tree_ensemble_models(df)
	run_hierarchical_clustering(df)

	print("\n🎉 Full analysis completed! Check the 'output/' folder.")
	print("Repository ready for submission + Docker bonus.")


if __name__ == "__main__":
	main()