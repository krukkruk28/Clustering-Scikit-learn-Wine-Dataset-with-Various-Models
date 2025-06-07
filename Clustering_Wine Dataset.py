import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
from sklearn.cluster import DBSCAN
import scipy.cluster.hierarchy as sch
import os
# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"
import keras
import torch
import torch.nn as nn
import torch.optim as optim
from keras import layers

wine = load_wine()

df_features = pd.DataFrame(wine.data, columns=wine.feature_names)
# Load the wine dataset
print(pd.DataFrame(df_features).head())

# Create a DataFrame for the target variable
df_target = pd.DataFrame(load_wine().target, columns=['target'])
print(df_target.value_counts())

num_features = len(wine.feature_names)
colors = ['red', 'green', 'blue']

class KMeansClusteringwithPCA:
    def __init__(self, features, target, n_clusters=3, n_components=2):
        self.features = features
        self.target = target
        self.n_clusters = n_clusters
        self.n_components = n_components
        self.standardized_data = None
        self.pca_data = None
        self.kmeans_labels = None

    def standardize_data(self):
        """Standardizes the dataset."""
        scaler = StandardScaler()
        self.standardized_data = scaler.fit_transform(self.features)

    def apply_pca(self):
        """Applies PCA for dimensionality reduction."""
        pca = PCA(n_components=self.n_components)
        self.pca_data = pca.fit_transform(self.standardized_data)

    def perform_kmeans(self):
        """Performs KMeans clustering."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.pca_data)
        self.kmeans_labels = kmeans.labels_

    def calculate_silhouette_score(self, data, labels):
        """Computes silhouette score."""
        return silhouette_score(data, labels)

    def visualize_elbow_method(self, max_clusters=10):
        """Plots elbow method to determine the optimal number of clusters."""
        inertia = []
        for n in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=42)
            kmeans.fit(self.standardized_data)
            inertia.append(kmeans.inertia_)
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_clusters + 1), inertia, marker='o')
        plt.title('Elbow Method for Optimal K')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Inertia')
        plt.grid()
        plt.show()

    def visualize_clusters(self, data, labels, title="Clustering Visualization"):
        """Plots clusters."""
        plt.figure(figsize=(10, 6))
        plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel('Principal Component 1' if title.endswith("(PCA Reduced)") else "Feature 1")
        plt.ylabel('Principal Component 2' if title.endswith("(PCA Reduced)") else "Feature 2")
        plt.colorbar(label='Cluster Label')
        plt.show()

    def visualize_hierarchical_clustering(self):
        """Plots hierarchical clustering dendrogram."""
        linkage_matrix = sch.linkage(self.standardized_data, method='complete')
        plt.figure(figsize=(8, 5))
        sch.dendrogram(linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.show()

    def execute(self):
        """Runs the entire clustering workflow."""
        self.standardize_data()
        self.apply_pca()
        self.perform_kmeans()
        
        # Print silhouette scores
        print(f"Silhouette Score (PCA vs KMeans): {self.calculate_silhouette_score(self.pca_data, self.kmeans_labels):.2f}")
        print(f"Silhouette Score (PCA vs Target): {self.calculate_silhouette_score(self.pca_data, self.target):.2f}")
        print(f"Silhouette Score (Standardized vs KMeans): {self.calculate_silhouette_score(self.standardized_data, self.kmeans_labels):.2f}")
        print(f"Silhouette Score (Standardized vs Target): {self.calculate_silhouette_score(self.standardized_data, self.target):.2f}")

        # Visualizations
        self.visualize_elbow_method()
        self.visualize_clusters(self.pca_data, self.kmeans_labels, "KMeans Clustering (PCA Reduced)")
        self.visualize_clusters(self.standardized_data, self.kmeans_labels, "KMeans Clustering (Original Features)")
        self.visualize_hierarchical_clustering()

class DBSCANClustering:
    def __init__(self, features, target, eps=0.5, min_samples=5, n_components=2):
        self.features = features
        self.target = target
        self.eps = eps
        self.min_samples = min_samples
        self.n_components = n_components
        self.standardized_data = None
        self.pca_data = None
        self.dbscan_labels = None

    def standardize_data(self):
        """Standardizes the dataset."""
        scaler = StandardScaler()
        self.standardized_data = scaler.fit_transform(self.features)

    def apply_pca(self):
        """Applies PCA for dimensionality reduction."""
        pca = PCA(n_components=self.n_components)
        self.pca_data = pca.fit_transform(self.standardized_data)

    def perform_dbscan(self):
        """Performs DBSCAN clustering."""
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.dbscan_labels = dbscan.fit_predict(self.pca_data)

    def calculate_silhouette_score(self):
        """Computes silhouette score for DBSCAN clusters."""
        # Ensure valid labels (DBSCAN assigns -1 to noise)
        unique_labels = set(self.dbscan_labels)
        if len(unique_labels - {-1}) > 1:  # At least 2 clusters
            return silhouette_score(self.pca_data, self.dbscan_labels)
        return None  # Silhouette Score not valid for 1 cluster

    def visualize_clusters(self, title="DBSCAN Clustering"):
        """Plots clusters."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.pca_data[:, 0], self.pca_data[:, 1], c=self.dbscan_labels, cmap='viridis', marker='o', edgecolor='k', s=50)
        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster Label")
        plt.show()

    def visualize_hierarchical_clustering(self):
        """Plots hierarchical clustering dendrogram (optional for comparison)."""
        linkage_matrix = sch.linkage(self.standardized_data, method='complete')
        plt.figure(figsize=(8, 5))
        sch.dendrogram(linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        plt.show()

    def execute(self):
        """Runs the entire DBSCAN clustering workflow."""
        self.standardize_data()
        self.apply_pca()
        self.perform_dbscan()

        # Compute silhouette score (if valid)
        silhouette_score_value = self.calculate_silhouette_score()
        if silhouette_score_value:
            print(f"Silhouette Score (PCA vs DBSCAN): {silhouette_score_value:.2f}")
        else:
            print("Silhouette Score: Not applicable (only 1 cluster detected)")

        # Visualizations
        self.visualize_clusters()
        self.visualize_hierarchical_clustering()

class KerasClustering:
    def __init__(self, features, target, encoding_dim=10, n_clusters=3):
        self.features = features
        self.target = target
        self.encoding_dim = encoding_dim
        self.n_clusters = n_clusters
        self.standardized_data = None
        self.encoded_data = None
        self.kmeans_labels = None
        self.autoencoder = None

    def standardize_data(self):
        """Standardizes the dataset."""
        scaler = StandardScaler()
        self.standardized_data = scaler.fit_transform(self.features)

    def build_autoencoder(self):
        """Builds a standalone Keras autoencoder using NumPy backend."""
        input_dim = self.features.shape[1]
        input_layer = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(self.encoding_dim, activation="relu")(input_layer)
        decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)

        self.autoencoder = keras.Model(inputs=input_layer, outputs=decoded)
        encoder = keras.Model(inputs=input_layer, outputs=encoded)

        self.autoencoder.compile(optimizer="adam", loss="mse")
        return encoder

    def train_autoencoder(self, epochs=50, batch_size=16):
        """Trains the autoencoder."""
        encoder = self.build_autoencoder()
        self.autoencoder.fit(self.standardized_data, self.standardized_data, epochs=epochs, batch_size=batch_size, verbose=0)
        self.encoded_data = encoder.predict(self.standardized_data)

    def perform_kmeans(self):
        """Performs K-Means clustering on encoded data."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.encoded_data)
        self.kmeans_labels = kmeans.labels_

    def calculate_silhouette_score(self):
        """Computes silhouette score."""
        return silhouette_score(self.encoded_data, self.kmeans_labels)

    def visualize_clusters(self):
        """Plots clusters."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.encoded_data[:, 0], self.encoded_data[:, 1], c=self.kmeans_labels, cmap="viridis", edgecolor="k", s=50)
        plt.title("KMeans Clustering with Autoencoder Features (Standalone Keras)")
        plt.xlabel("Encoded Feature 1")
        plt.ylabel("Encoded Feature 2")
        plt.colorbar(label="Cluster Label")
        plt.show()

    def execute(self):
        """Runs the entire clustering workflow."""
        self.standardize_data()
        self.train_autoencoder()
        self.perform_kmeans()

        print(f"Silhouette Score of Keras is: {self.calculate_silhouette_score():.2f}")
        self.visualize_clusters()

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class PyTorchClustering:
    def __init__(self, features, target, encoding_dim=10, n_clusters=3, epochs=50, batch_size=16, lr=0.01):
        self.features = features
        self.target = target
        self.encoding_dim = encoding_dim
        self.n_clusters = n_clusters
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.standardized_data = None
        self.encoded_data = None
        self.kmeans_labels = None
        self.autoencoder = None

    def standardize_data(self):
        """Standardizes the dataset."""
        scaler = StandardScaler()
        self.standardized_data = scaler.fit_transform(self.features)

    def train_autoencoder(self):
        """Trains an autoencoder using PyTorch."""
        input_dim = self.features.shape[1]
        self.autoencoder = Autoencoder(input_dim, self.encoding_dim)
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Convert data to PyTorch tensors
        data_tensor = torch.tensor(self.standardized_data, dtype=torch.float32)

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            encoded, decoded = self.autoencoder(data_tensor)
            loss = criterion(decoded, data_tensor)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch [{epoch}/{self.epochs}] Loss: {loss.item():.4f}")

        # Extract encoded features
        self.encoded_data = encoded.detach().numpy()

    def perform_kmeans(self):
        """Performs K-Means clustering on encoded data."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans.fit(self.encoded_data)
        self.kmeans_labels = kmeans.labels_

    def calculate_silhouette_score(self):
        """Computes silhouette score."""
        return silhouette_score(self.encoded_data, self.kmeans_labels)

    def visualize_clusters(self):
        """Plots clusters."""
        plt.figure(figsize=(10, 6))
        plt.scatter(self.encoded_data[:, 0], self.encoded_data[:, 1], c=self.kmeans_labels, cmap="viridis", edgecolor="k", s=50)
        plt.title("KMeans Clustering with Autoencoder Features (PyTorch)")
        plt.xlabel("Encoded Feature 1")
        plt.ylabel("Encoded Feature 2")
        plt.colorbar(label="Cluster Label")
        plt.show()

    def execute(self):
        """Runs the entire clustering workflow."""
        self.standardize_data()
        self.train_autoencoder()
        self.perform_kmeans()

        print(f"Silhouette Score of Pytorch is: {self.calculate_silhouette_score():.2f}")
        self.visualize_clusters()

# # Using KMeans Clustering
# clustering = KMeansClusteringwithPCA(df_features, df_target['target'])
# clustering.execute()

# # Using DBSCAN
# clustering = DBSCANClustering(df_features, df_target['target'])
# clustering.execute()

# # Using Keras Clustering
# clustering = KerasClustering(df_features, df_target['target'])
# clustering.execute()

# # Using Pytorch Clustering
# clustering = PyTorchClustering(df_features, df_target['target'])
# clustering.execute()

"""
Documentation:
This repository explores multiple clustering techniques to evaluate their accuracy across different models. The goal is to compare their effectiveness in various scenarios and determine the best approach for the given dataset.

Conclusion:
KMeans Clustering with PCA applied has 0.56 silhouette score, demonstrating strong clustering performance.
Deep learning models reached a silhouette score of 0.44, indicating moderate separation between clusters.

Therefore, KMeans performed better on small datasets, suggesting its suitability for compact feature spaces. Future work will focus on applying deep learning models to larger datasets, where their advanced feature extraction capabilities may provide better clustering outcomes.

"""
