import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(df):
    print("🔍 Running Exploratory Data Analysis...")

    # Ensure output folder exists and required columns are present (avoid KeyError when columns missing)
    os.makedirs('output', exist_ok=True)
    required_cols = ['MEDIAN_PRICE', 'VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT',
                     'DRUG_CRIME_COUNT', 'OTHER_CRIME_COUNT', 'SHOOTING_COUNT']
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0

    # Histograms with KDE
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    sns.histplot(df['MEDIAN_PRICE'], kde=True, ax=axes[0,0]).set(title='Median Price Distribution')
    sns.histplot(df['VIOLENT_CRIME_COUNT'], kde=True, ax=axes[0,1]).set(title='Violent Crime')
    sns.histplot(df['PROPERTY_CRIME_COUNT'], kde=True, ax=axes[0,2]).set(title='Property Crime')
    sns.histplot(df['DRUG_CRIME_COUNT'], kde=True, ax=axes[1,0]).set(title='Drug Crime')
    sns.histplot(df['OTHER_CRIME_COUNT'], kde=True, ax=axes[1,1]).set(title='Other Crime')
    sns.histplot(df['SHOOTING_COUNT'], kde=True, ax=axes[1,2]).set(title='Shootings')
    plt.tight_layout()
    plt.savefig('output/eda_distributions.png')
    plt.close()

    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    corr = df[['MEDIAN_PRICE', 'VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT',
               'DRUG_CRIME_COUNT', 'OTHER_CRIME_COUNT', 'SHOOTING_COUNT']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap: Crime Types vs Median Price')
    plt.savefig('output/correlation_heatmap.png')
    plt.close()

    print("✅ EDA plots saved to output/ folder")