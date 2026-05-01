import pandas as pd
from scipy import stats
import numpy as np

def run_hypothesis_test(df):
    print("\n📊 Hypothesis Testing - Pearson Correlation")
    for col in ['VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT', 'DRUG_CRIME_COUNT',
                'OTHER_CRIME_COUNT', 'SHOOTING_COUNT']:
        corr, p = stats.pearsonr(df[col], np.log1p(df['MEDIAN_PRICE']))
        print(f"{col:22} → r = {corr:.4f}, p-value = {p:.6f}")