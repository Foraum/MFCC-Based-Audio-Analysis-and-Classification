import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Define the path to your MFCC CSV files
mfcc_path = "MFCC-files-v2-20241024"  # Update with your directory path

# Step 1: Load and Preprocess Data
def load_mfcc_files(path):
    data = []
    filenames = []
    for filename in os.listdir(path):
        if filename.endswith("-MFCC.csv"):
            song_id = filename.split('-')[0]  # Extract song number
            file_path = os.path.join(path, filename)
            mfcc_data = pd.read_csv(file_path, header=None)
            print(file_path)
            
            # Aggregate MFCC statistics (mean, std, skewness) for each coefficient
            mfcc_features = []
            for col in mfcc_data.columns:
                mfcc_features.extend([
                    mfcc_data[col].mean(),
                    mfcc_data[col].std(),
                    mfcc_data[col].skew()
                ])
            
            data.append(mfcc_features)
            filenames.append(song_id)
    return pd.DataFrame(data, index=filenames)

# Load data
mfcc_df = load_mfcc_files(mfcc_path)
print("Data shape:", mfcc_df.shape)

# Step 2: Feature Scaling
scaler = StandardScaler()
mfcc_scaled = scaler.fit_transform(mfcc_df)

# Step 3: Dimensionality Reduction with PCA
pca = PCA(n_components=10)  # Adjust components if necessary
mfcc_pca = pca.fit_transform(mfcc_scaled)
print("PCA explained variance ratio:", pca.explained_variance_ratio_.sum())

# Step 4: Clustering
kmeans = KMeans(n_clusters=6, random_state=42)  # Adjust cluster count as needed
clusters = kmeans.fit_predict(mfcc_pca)
mfcc_df['Cluster'] = clusters

# Step 5: t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
mfcc_tsne = tsne.fit_transform(mfcc_pca)

# Plotting
plt.figure(figsize=(10, 8))
sns.scatterplot(x=mfcc_tsne[:, 0], y=mfcc_tsne[:, 1], hue=clusters, palette="viridis", s=60)
plt.title("t-SNE Visualization of Clusters")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(title="Cluster")
plt.show()

# Step 6: Summary Table
# Print cluster membership for each song
for cluster in sorted(mfcc_df['Cluster'].unique()):
    print(f"Cluster {cluster}:")
    print(mfcc_df[mfcc_df['Cluster'] == cluster].index.values)
