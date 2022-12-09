import pandas as pd
import numpy as np

def dist(p1: list, p2: list):
    """Calculates the Euclidean distance between two points in n-dimensional space"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list"""
    return cluster_df.describe().loc['mean']

def intra_cluster_variance(cluster: pd.DataFrame) -> float:
    centroid = calculate_centroid(cluster)
    [dist(instance, centroid) for instance in cluster.values.tolist()]
    return np.sum()

def WCSS(cluster: list):
    """Calculate the Within-Cluster-Sum of Squared-Errors (WSS)
    WCSS is the sum of squares of the distances of each data 
    point in all clusters to their respective centroids"""
    
    n_k = len(cluster)