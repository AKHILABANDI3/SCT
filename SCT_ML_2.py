import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 1: Load the dataset
data = pd.read_csv('Mall_Customers.csv')

# Step 2: Select relevant features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Step 3: Apply K-Means clustering (using k=5, common for this dataset)
kmeans = KMeans(n_clusters=5, random_state=42)
data['Cluster'] = kmeans.fit_predict(X)

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="Set1",
    data=data,
    s=70,
)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    c='black',
    s=200,
    marker='X',
    label='Centroids'
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("K-means Customer Segments")
plt.legend()
plt.tight_layout()
plt.show()

# Step 5: Output basic cluster statistics
print("\nCluster Centers (Annual Income / Spending Score):")
for idx, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {idx}: {center}")

print("\nCluster Sizes:")
print(data['Cluster'].value_counts())

print("\nCluster Means:")
print(data.groupby('Cluster')[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean())
