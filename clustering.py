from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

def run_hierarchical_clustering(df):
    print("\n🔬 Running Hierarchical Clustering...")
    cluster_features = ['VIOLENT_CRIME_COUNT', 'PROPERTY_CRIME_COUNT', 'DRUG_CRIME_COUNT',
                        'OTHER_CRIME_COUNT', 'SHOOTING_COUNT', 'MEDIAN_PRICE']
    X_cluster = df[cluster_features].copy()
    X_scaled = StandardScaler().fit_transform(X_cluster)

    Z = linkage(X_scaled, method='ward')

    plt.figure(figsize=(14, 7))
    dendrogram(Z, labels=[f"{b}-{int(y)}" for b, y in zip(df['BOROUGH'], df['YEAR'])],
               leaf_rotation=90, leaf_font_size=11)
    plt.title('Hierarchical Clustering Dendrogram - NYC Borough-Year Safety Zones')
    plt.xlabel('Borough-Year')
    plt.ylabel('Distance (Ward Linkage)')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("output/dendrogram_safety_zones.png")
    plt.close()

    print("✅ Dendrogram saved to output/dendrogram_safety_zones.png")