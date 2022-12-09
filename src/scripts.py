import pandas as pd
import numpy as np

def dist(p1: list, p2: list):
    """Calculates the Euclidean distance between two points in n-dimensional space"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list"""
    return cluster_df.describe().loc['mean']

def split_in_clusters(cluster_df: pd.DataFrame) -> list:
    cluster_list = [] # a list to hold each cluster as a DataFrame
    for i in range(len(cluster_df.cluster.unique())):
        cluster_list.append(cluster_df.loc[cluster_df['cluster'] == i])
    return cluster_list

def intra_cluster_variance(cluster_df: pd.DataFrame) -> float:  
    cluster_list = split_in_clusters(cluster_df)
    
    # calculate centroid in each cluster
    cluster_centroids = [calculate_centroid(cluster) for cluster in cluster_list]
    
    # calculate intra variance in each cluster and store in list
    cluster_intra_variance = []
    for (cluster, centroid) in zip(cluster_list, cluster_centroids):
        cluster_intra_variance.append(sum([dist(instance, centroid) for instance in cluster.values.tolist()]))
    
    return cluster_list, cluster_intra_variance
    # A 2D list of distances between each instance and its centroid in each cluster
    #cluster_centroid_distances = [[dist(instance, centroid) for instance in cluster.values.tolist()] for cluster in cluster_list]
    
    #return cluster_centroid_distances # np.sum(intra_variances)

def WCSS(cluster: list):
    """Calculate the Within-Cluster-Sum of Squared-Errors (WSS)
    WCSS is the sum of squares of the distances of each data 
    point in all clusters to their respective centroids"""
    
    n_k = len(cluster)