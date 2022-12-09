import pandas as pd
import numpy as np

def dist(p1: list, p2: list):
    """Calculates the Euclidean distance between two points in n-dimensional space"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list"""
    return cluster_df.describe().loc['mean']

def intra_cluster_variance(cluster: pd.DataFrame) -> float:
    """Calculate the Intra-Cluster-Variance (ICV) of a cluster
    ICV is the sum of squares of the distances of each data
    point in a cluster to its centroid"""
    centroid = calculate_centroid(cluster)
    [dist(instance, centroid) for instance in cluster.values.tolist()]
    return np.sum()
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

def evaluate_clusters_plot(df: pd.DataFrame, startrange = 2, stoprange = 25, cluster_method = KMeans()) -> None:
    '''
    Evaluate clusters using the Davies Bouldin Score, Silhouette Score and Calinski Harabasz Score.
    Plot the results for each metric.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to be used for clustering evaluation
    startrange : int, optional
        Starting number of clusters. The default is 2.
    stoprange : int, optional
        Stopping number of clusters. The default is 25.
    cluster_method : sklearn.cluster, optional
        Clustering method to be used. The default is KMeans.
        Another example: cluster_method = AgglomerativeClustering.

    Returns
    -------
    None.
    '''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,6))
    resultsa = {}
    resultsb = {}
    resultsc = {}
    
    for i in range(startrange, stoprange):
        cluster = cluster_method
        cluster = cluster(n_clusters=i)
        labels = cluster.fit_predict(df)
        labels = cluster.labels_
        
        db_index = davies_bouldin_score(df, labels)
        resultsa.update({i: db_index})
        db_index = silhouette_score(df, labels)
        resultsb.update({i: db_index})
        db_index = calinski_harabasz_score(df, labels)
        resultsc.update({i: db_index})

    fig.suptitle(f'Evaluation Metrics for {str(cluster_method())[:-2]}', fontsize=15)
    ax1.plot(list(resultsa.keys()), list(resultsa.values()), marker='o')
    ax1.set_title('Davies Bouldin Score', fontsize=15)
    ax2.plot(list(resultsb.keys()), list(resultsb.values()), marker='o')
    ax2.set_title('Silhouette Score', fontsize=15)
    ax3.plot(list(resultsc.keys()), list(resultsc.values()), marker='o')
    ax3.set_title('Calinski Harabasz Score', fontsize=15)

