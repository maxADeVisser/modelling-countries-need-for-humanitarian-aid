import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def dist(p1: list, p2: list):
    """Calculates the Euclidean distance between 
    two points in n-dimensional space"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))


def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list"""
    return cluster_df.describe().loc['mean']


def ICV(cluster: pd.DataFrame) -> float:  
    """Calculate the Intra-Cluster-Variance (ICV) of the provided cluster."""
    
    # calculate n-dimensional centroid for cluster
    cluster_centroid = calculate_centroid(cluster)
    
    distances_to_centroid = [dist(instance, cluster_centroid) for instance in cluster.values.tolist()]
    return np.mean(distances_to_centroid)


def split_in_clusters(cluster_df: pd.DataFrame) -> list:
    result = {}
    for i in range(len(cluster_df.cluster.unique())):
        result[i] = cluster_df.loc[cluster_df['cluster'] == i]
    return result

def evalutate_clusters(clustered_df: pd.DataFrame):
    """Calculate the silhouette score, Calinski-Harabasz Index, Davies-Bouldin Index of a clustered dataframe (in that order).
    The dataframe needs to have a 'cluster' column, and the rest of the columns are the features."""
    s = silhouette_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    c = calinski_harabasz_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    d = davies_bouldin_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    return s, c, d


# def WCSS(cluster: list):
#     """Calculate the Within-Cluster-Sum of Squared-Errors (WSS)
#     WCSS is the sum of squares of the distances of each data 
#     point in all clusters to their respective centroids"""
    
#     n_k = len(cluster)