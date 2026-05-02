# Urban Pulse: Neighborhood Safety & NYC Real Estate Prices

**Team:** Ayonitemi Bajimilehin, Hein Vo, Emmanuel Cruzat, Joshua Harrison

## Problem Statement
We investigated a core **Urban Pulse** question:  
**To what extent do the volume and severity of reported crimes (NYPD Arrests + Shootings) affect median residential property sale prices in New York City boroughs?**

**Hypotheses**  
- **H₀**: No statistically significant relationship between crime volume/severity and median property prices.  
- **Hₐ**: There is a statistically significant **negative** relationship.

## Data Sources & Linking Strategy
All data was loaded directly via **NYC OpenData Socrata API** (no full downloads required):
- NYPD Arrests (2010–2023)
- NYPD Shootings (historic)
- NYC Housing Sales (2010–2023, residential only)

**Linking method**: Aggregated at the **Borough + Year** level (natural foreign-key join).  
Crimes were categorized into `VIOLENT`, `PROPERTY`, `DRUG`, `OTHER`; shootings kept as a separate high-severity indicator.

## Descriptive Statistics Overview
- ~25 borough-year observations after aggregation.
- Median Price ranges from ~$425k (Bronx) to >$1.2M (Manhattan).
- Crime counts are heavily right-skewed — most borough-years have low/zero values with rare high-crime outliers.
- Manhattan and Brooklyn consistently show the highest prices **and** highest absolute crime volumes (density effect).

## Exploratory Data Analysis

### Feature Distributions (6 Histograms + KDE)
All variables (`MEDIAN_PRICE`, `VIOLENT_CRIME_COUNT`, `PROPERTY_CRIME_COUNT`, `DRUG_CRIME_COUNT`, `OTHER_CRIME_COUNT`, `SHOOTING_COUNT`) are **strongly right-skewed** with long right tails.  
- X-axis: actual values of the variable.  
- Y-axis: Count (number of borough-year observations in each bin).  
- KDE curves confirm non-normality → justified log transformation on target and use of robust models.

### Correlation Heatmap
The center is heavily **red/orange** (positive correlations +0.25 to +0.39) for most crime types with `MEDIAN_PRICE`.  
**Only `SHOOTING_COUNT` shows a negative correlation** (r ≈ -0.12, blue cell).  
This is classic confounding by borough: high-price areas naturally have higher crime counts due to density.

## Hypothesis Testing
Pearson correlations show most p-values > 0.05 in raw data (limited by confounding). `SHOOTING_COUNT` consistently has the most negative r and lowest p-value.

## Supervised Learning Results

### Regression Models
- Polynomial Degree 1 (scaled linear regression) performed best in the latest run.
- Gradient Boosting and Random Forest were strong performers and provided excellent feature importance.
- Top features: `BOROUGH_M` (~0.52) and `BOROUGH_K` (~0.23) dominate; crime features contribute meaningfully after controlling for borough.

### Classification & Tree-based Models
- Decision Trees (CART) clearly show **borough first**, then **PROPERTY_CRIME_COUNT** (threshold ≈ 374) as the first crime split.
- **Full (deep) tree**: Detailed but risks overfitting.
- **Pruned tree** (`max_depth=4`, `min_samples_split=5`): Much cleaner, more interpretable, and generalizable.

### K-Fold Cross Validation
Confirmed robust performance of linear and ensemble models.

## Unsupervised Learning: Hierarchical Clustering + Dendrogram
Ward linkage on standardized crime + price vectors produced clear **Safety Zones**.  
The dendrogram shows a very tall merge (height ~12–13) separating the **2021 borough-years** as a distinct cluster — a clear pandemic-recovery outlier.

## Within-Neighborhood Analysis (Granular View)
By fixing the neighborhood and comparing years within it, crime explains 70–80% of price variation in several real neighborhoods (e.g., Downtown-Fulton Ferry, Ocean Parkway-North, Harlem-Upper, Bay Ridge, Roosevelt Island).  
This is stronger evidence than borough-level analysis because location is held constant.

## Knowledge Discovery (The "Aha!" Moment)
Multiple techniques converge on the same story:
- **Borough dominates**, but **crime still matters** once location is controlled for.
- **Property crime** has the clearest threshold effect (first crime split at ≈374 incidents in pruned tree).
- **Shootings** are the only crime type with a negative raw correlation and remain important across models.
- **2021** is a unique outlier year (visible in dendrogram and tree splits).
- **Surprising result**: Raw correlations are mostly positive (confounding), but models and within-neighborhood regressions reveal the true negative crime-price relationship.

## Actionable Business & Policy Insights

**Significant impacting factors**:
1. Location (borough) is the strongest driver.
2. Property crime has a clear negative threshold effect (~374 incidents).
3. Violent crime and shootings exert additional downward pressure.
4. 2021 was a distinct high-sensitivity year.

**Recommended Solutions**:
- **City Government / NYPD**: Prioritize violent and property crime reduction in Brooklyn and Bronx. Use the pruned tree threshold (~374 property crimes) as a targeting guideline.
- **Real Estate Investors**: Screen neighborhoods using the decision-tree rule — avoid or discount borough-years with high property crime **and** elevated shootings (unless in Manhattan). Pay special attention to sensitive neighborhoods like Harlem-Upper, Bay Ridge, Roosevelt Island.
- **Economic Development Agencies**: Treat safety improvements as high-ROI economic policy. Reducing shootings and property crime directly translates into higher residential values, increased tax revenue, and stronger business investment confidence.
- **Overall Recommendation**: Combine policing, community programs, and urban planning to keep crime below critical thresholds. Neighborhood safety is a direct, quantifiable economic driver of real estate value in New York City.

## How to Run
```bash
pip install -r requirements.txt
python main.py
