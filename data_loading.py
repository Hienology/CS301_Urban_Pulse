import pandas as pd

def load_and_aggregate_data():
    print("📥 Loading data from NYC OpenData Socrata API...")

    # NYPD Arrests
    arrests_url = "https://data.cityofnewyork.us/resource/8h9b-rp9u.csv?$select=arrest_date,arrest_boro,pd_cd,ky_cd,ofns_desc&$where=arrest_date>='2010-01-01'+AND+arrest_date<'2024-01-01'&$limit=500000"
    arrests = pd.read_csv(arrests_url, low_memory=False)
    arrests['ARREST_DATE'] = pd.to_datetime(arrests['arrest_date'], errors='coerce')
    arrests['YEAR'] = arrests['ARREST_DATE'].dt.year
    arrests['BOROUGH'] = arrests['arrest_boro']

    def categorize_crime(row):
        desc = str(row['ofns_desc']).upper() if pd.notna(row['ofns_desc']) else ""
        if any(x in desc for x in ['MURDER','HOMICIDE','RAPE','ROBBERY','ASSAULT','FELONY ASSAULT']):
            return 'VIOLENT'
        elif any(x in desc for x in ['BURGLARY','LARCENY','THEFT','CRIMINAL MISCHIEF']):
            return 'PROPERTY'
        elif any(x in desc for x in ['DRUG','CONTROLLED SUBSTANCE','MARIHUANA']):
            return 'DRUG'
        return 'OTHER'

    arrests['CRIME_TYPE'] = arrests.apply(categorize_crime, axis=1)

    # NYPD Shootings
    shootings = pd.read_csv("https://data.cityofnewyork.us/resource/833y-fsy8.csv")
    shootings['OCCUR_DATE'] = pd.to_datetime(shootings['occur_date'], errors='coerce')
    shootings['YEAR'] = shootings['OCCUR_DATE'].dt.year
    boro_map = {'MANHATTAN':'M', 'BRONX':'B', 'BROOKLYN':'K', 'QUEENS':'Q', 'STATEN ISLAND':'S'}
    shootings['BOROUGH'] = shootings['boro'].map(boro_map)

    # NYC Housing Sales
    housing = pd.read_csv("https://data.cityofnewyork.us/resource/w2pb-icbu.csv?$select=borough,sale_date,sale_price,building_class_category&$where=sale_price>0+AND+sale_date>='2010-01-01'&$limit=200000", low_memory=False)
    housing['SALE_DATE'] = pd.to_datetime(housing['sale_date'], errors='coerce')
    housing['YEAR'] = housing['SALE_DATE'].dt.year
    housing['BOROUGH'] = housing['borough'].map({1:'M',2:'B',3:'K',4:'Q',5:'S'})

    # Aggregation
    arrests_agg = arrests.groupby(['YEAR', 'BOROUGH', 'CRIME_TYPE']).size().unstack(fill_value=0).reset_index()
    arrests_agg = arrests_agg.rename(columns={'VIOLENT':'VIOLENT_CRIME_COUNT', 'PROPERTY':'PROPERTY_CRIME_COUNT',
                                              'DRUG':'DRUG_CRIME_COUNT', 'OTHER':'OTHER_CRIME_COUNT'})

    shootings_agg = shootings.groupby(['YEAR', 'BOROUGH']).size().reset_index(name='SHOOTING_COUNT')

    residential = housing[housing['building_class_category'].str.contains('1|2|APARTMENT|RESIDENTIAL|COOP|CONDO|HOUSE', na=False, case=False)]
    housing_agg = residential.groupby(['YEAR', 'BOROUGH'])['sale_price'].median().reset_index(name='MEDIAN_PRICE')

    df = pd.merge(housing_agg, arrests_agg, on=['YEAR', 'BOROUGH'], how='left')
    df = pd.merge(df, shootings_agg, on=['YEAR', 'BOROUGH'], how='left')
    df = df.fillna(0)

    # Ensure expected columns exist so downstream code (EDA, models) won't KeyError
    expected_cols = ['MEDIAN_PRICE', 'VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT',
                     'DRUG_CRIME_COUNT', 'OTHER_CRIME_COUNT', 'SHOOTING_COUNT']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    df = df.sort_values(['YEAR', 'BOROUGH']).reset_index(drop=True)

    print("✅ Data loaded and aggregated successfully!")
    return df