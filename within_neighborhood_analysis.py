"""
within_neighborhood_analysis.py
Granular analysis: crime impact on prices WITHIN the same neighborhood over time.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def run_within_neighborhood_analysis(df, housing):
    """
    Runs within-neighborhood regression using existing df (crime aggregates) and housing data.
    """
    print("🔬 WITHIN-NEIGHBORHOOD ANALYSIS")
    print("Comparing crime impact on prices **within the same neighborhood** over time\n")

    # Aggregate median price by Neighborhood + Year
    neigh_price = housing[
        housing['building_class_category'].str.contains('1|2|APARTMENT|RESIDENTIAL|COOP|CONDO|HOUSE', na=False, case=False)
    ].groupby(['neighborhood', 'YEAR'])['sale_price'].median().reset_index(name='MEDIAN_PRICE')

    neigh_price = neigh_price.rename(columns={'neighborhood': 'NEIGHBORHOOD'})

    # Merge with crime aggregates (borough-year level as control)
    neigh_analysis = pd.merge(
        neigh_price,
        df[['YEAR', 'BOROUGH', 'VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT',
            'DRUG_CRIME_COUNT', 'OTHER_CRIME_COUNT', 'SHOOTING_COUNT']],
        on='YEAR',
        how='left'
    )

    print(f"Within-neighborhood dataset shape: {neigh_analysis.shape}")

    # Regression per neighborhood
    results_within = []
    for neigh, group in neigh_analysis.groupby('NEIGHBORHOOD'):
        if len(group) < 4:          # Need at least 4 years for meaningful regression
            continue
        X_neigh = group[['VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT', 'SHOOTING_COUNT']]
        y_neigh = np.log1p(group['MEDIAN_PRICE'])
        
        if len(X_neigh) > 1:
            model = LinearRegression().fit(X_neigh, y_neigh)
            r2 = model.score(X_neigh, y_neigh)
            coef_v = model.coef_[0]
            coef_p = model.coef_[1]
            coef_s = model.coef_[2]
            results_within.append([neigh, len(group), r2, coef_v, coef_p, coef_s])

    within_df = pd.DataFrame(results_within, columns=[
        'Neighborhood', 'Years_Available', 'R²', 'Violent_Coef', 
        'Property_Coef', 'Shooting_Coef'
    ])

    print("\n📊 Top 10 Neighborhoods by R² (Within-Neighborhood Regression)")
    print(within_df.sort_values('R²', ascending=False).head(15))

    within_df.to_csv('output/within_neighborhood_regression_results.csv', index=False)
    print("\n✅ Within-neighborhood analysis completed!")
    print("Results saved to: output/within_neighborhood_regression_results.csv")

    return within_df
