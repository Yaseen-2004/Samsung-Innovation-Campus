import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class WorldDevelopmentClustering:
    """
    ML Clustering mechanism for analyzing world development indicators.
    Supports multiple clustering algorithms and evaluation metrics.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the clustering object.
        
        Parameters:
        -----------
        data_path : str
            Path to CSV file containing world development data
        """
        self.data = None
        self.scaled_data = None
        self.scaler = StandardScaler()
        self.clusters = None
        self.algorithm_name = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load world development data from CSV file."""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            print(f"Columns: {self.data.columns.tolist()}")
            return self
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            return None
    
    def preprocess_data(self, numeric_cols=None, handle_missing='drop'):
        """
        Preprocess data: handle missing values and scale features.
        
        Parameters:
        -----------
        numeric_cols : list
            List of numeric columns to use for clustering
        handle_missing : str
            Strategy for missing values: 'drop' or 'mean'
        """
        if self.data is None:
            print("No data loaded. Use load_data() first.")
            return None
        
        # Select numeric columns
        if numeric_cols is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_for_clustering = self.data[numeric_cols].copy()
        
        # Handle missing values
        if handle_missing == 'drop':
            data_for_clustering = data_for_clustering.dropna()
        elif handle_missing == 'mean':
            data_for_clustering = data_for_clustering.fillna(data_for_clustering.mean())
        
        # Scale the data
        self.scaled_data = self.scaler.fit_transform(data_for_clustering)
        print(f"Data preprocessed! Shape after preprocessing: {self.scaled_data.shape}")
        
        return self
    
    def find_optimal_k(self, k_range=range(2, 11), method='elbow'):
        """
        Find optimal number of clusters.
        
        Parameters:
        -----------
        k_range : range
            Range of k values to test
        method : str
            Method to use: 'elbow', 'silhouette', or 'both'
        """
        if self.scaled_data is None:
            print("Data not preprocessed. Use preprocess_data() first.")
            return None
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.scaled_data, kmeans.labels_))
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow curve
        axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
        axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Silhouette scores
        axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].set_title('Silhouette Score for Different k', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal k (based on Silhouette Score): {optimal_k}")
        
        return optimal_k, inertias, silhouette_scores
    
    def kmeans_clustering(self, n_clusters=3):
        """
        Perform K-Means clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        """
        if self.scaled_data is None:
            print("Data not preprocessed. Use preprocess_data() first.")
            return None
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clusters = kmeans.fit_predict(self.scaled_data)
        self.algorithm_name = "K-Means"
        
        print(f"\nK-Means Clustering completed with {n_clusters} clusters")
        self._print_evaluation_metrics()
        
        return self
    
    def hierarchical_clustering(self, n_clusters=3, linkage='ward'):
        """
        Perform Hierarchical (Agglomerative) clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage : str
            Linkage method: 'ward', 'complete', 'average', 'single'
        """
        if self.scaled_data is None:
            print("Data not preprocessed. Use preprocess_data() first.")
            return None
        
        hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.clusters = hierarchical.fit_predict(self.scaled_data)
        self.algorithm_name = f"Hierarchical ({linkage})"
        
        print(f"\nHierarchical Clustering completed with {n_clusters} clusters (linkage={linkage})")
        self._print_evaluation_metrics()
        
        return self
    
    def dbscan_clustering(self, eps=0.5, min_samples=5):
        """
        Perform DBSCAN clustering (density-based).
        
        Parameters:
        -----------
        eps : float
            Maximum distance between samples
        min_samples : int
            Minimum number of samples in a neighborhood
        """
        if self.scaled_data is None:
            print("Data not preprocessed. Use preprocess_data() first.")
            return None
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.clusters = dbscan.fit_predict(self.scaled_data)
        self.algorithm_name = "DBSCAN"
        
        n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        n_noise = list(self.clusters).count(-1)
        
        print(f"\nDBSCAN Clustering completed")
        print(f"Number of clusters: {n_clusters}")
        print(f"Number of noise points: {n_noise}")
        self._print_evaluation_metrics()
        
        return self
    
    def _print_evaluation_metrics(self):
        """Calculate and print clustering evaluation metrics."""
        if self.clusters is None:
            return
        
        # Silhouette Score
        silhouette = silhouette_score(self.scaled_data, self.clusters)
        print(f"Silhouette Score: {silhouette:.4f} (range: -1 to 1, higher is better)")
        
        # Davies-Bouldin Index
        davies_bouldin = davies_bouldin_score(self.scaled_data, self.clusters)
        print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        
        # Calinski-Harabasz Score
        calinski_harabasz = calinski_harabasz_score(self.scaled_data, self.clusters)
        print(f"Calinski-Harabasz Score: {calinski_harabasz:.4f} (higher is better)")
    
    def visualize_clusters_2d(self):
        """Visualize clusters in 2D using PCA."""
        if self.clusters is None:
            print("No clusters found. Run a clustering algorithm first.")
            return None
        
        # PCA for 2D visualization
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.scaled_data)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], 
                            c=self.clusters, cmap='viridis', 
                            s=100, alpha=0.6, edgecolors='k')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
        plt.title(f'{self.algorithm_name} Clustering (2D PCA Visualization)', 
                 fontsize=14, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return pca_data
    
    def visualize_clusters_3d(self):
        """Visualize clusters in 3D using PCA."""
        if self.clusters is None:
            print("No clusters found. Run a clustering algorithm first.")
            return None
        
        from mpl_toolkits.mplot3d import Axes3D
        
        # PCA for 3D visualization
        pca = PCA(n_components=3)
        pca_data = pca.fit_transform(self.scaled_data)
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2],
                           c=self.clusters, cmap='viridis', 
                           s=100, alpha=0.6, edgecolors='k')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})', fontsize=10)
        ax.set_title(f'{self.algorithm_name} Clustering (3D PCA Visualization)', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax, label='Cluster', shrink=0.5)
        
        plt.tight_layout()
        plt.show()
        
        return pca_data
    
    def get_cluster_summary(self):
        """Get summary statistics for each cluster."""
        if self.clusters is None:
            print("No clusters found. Run a clustering algorithm first.")
            return None
        
        self.data['Cluster'] = self.clusters
        summary = self.data.groupby('Cluster').size()
        print(f"\nCluster Distribution:\n{summary}")
        
        return summary


# Example Usage
if __name__ == "__main__":
    # Initialize clustering object
    clustering = WorldDevelopmentClustering()
    
    # Create sample world development data
    np.random.seed(42)
    n_countries = 100
    
    sample_data = pd.DataFrame({
        'Country': [f'Country_{i}' for i in range(n_countries)],
        'GDP_per_capita': np.random.uniform(500, 80000, n_countries),
        'Life_Expectancy': np.random.uniform(50, 85, n_countries),
        'Literacy_Rate': np.random.uniform(30, 99, n_countries),
        'HDI': np.random.uniform(0.3, 0.95, n_countries),
        'Population_Growth': np.random.uniform(-1, 3, n_countries)
    })
    
    clustering.data = sample_data
    
    # Preprocess data
    clustering.preprocess_data(
        numeric_cols=['GDP_per_capita', 'Life_Expectancy', 'Literacy_Rate', 'HDI', 'Population_Growth'],
        handle_missing='mean'
    )
    
    # Find optimal k
    optimal_k, inertias, silhouette_scores = clustering.find_optimal_k(k_range=range(2, 11))
    
    # Perform K-Means clustering
    clustering.kmeans_clustering(n_clusters=optimal_k)
    
    # Visualizations
    clustering.visualize_clusters_2d()
    # clustering.visualize_clusters_3d()  # Uncomment for 3D visualization
    
    # Get cluster summary
    clustering.get_cluster_summary()

