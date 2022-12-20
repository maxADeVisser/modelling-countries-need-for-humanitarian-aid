import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import geopandas
import folium
from datetime import datetime
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dist(p1: list, p2: list):
    """Calculates the Euclidean distance between 
    two points in n-dimensional space"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))


def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list"""
    return cluster_df.describe().loc['mean']


def ICV(cluster: pd.DataFrame) -> float:  
    """Calculate the Intra-Cluster-Variance (ICV) of the provided cluster.
    This is calculated as the mean of the distances of each data point in a cluster,
    to every other data point in the same cluster."""
    
    average_distances = []
    
    for sample in cluster.values.tolist():
        current = sample
        
        distances_from_current = []
        for point in cluster.values.tolist():
            if current != point:
                distances_from_current.append(dist(current, point))
        average_distances.append(np.mean(distances_from_current))

    return average_distances


def split_in_clusters(cluster_df: pd.DataFrame) -> list:
    """Returns a dict with the clusters as values and the cluster number as key"""
    result = {} 
    for i in range(len(cluster_df.cluster.unique())):
        result[i] = cluster_df.loc[cluster_df['cluster'] == i]\
            .drop(columns = ['cluster'], axis=1)
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
    
    #return cluster_centroid_distances # np.sum(intra_variances)

def pre_process_data(data: pd.DataFrame, scaler: str = 'standard', pca = False, pca_components: int = 9, plot_scree_plot: bool = False):
    """Make into a function that can be imported and perform all pre-processing steps
    on the data. This includes scaling, PCA, etc.

    
    Parameters
    ----------
    data : pd.DataFrame
        The data to be pre-processed
    scaler : str, optional
        The scaler to be used, by default 'standard'
        alternatives: {'minmax', 'robust', 'normalizer', 'quantile', 'power', 'maxabs', 'function'}
    pca : bool, optional
        Whether or not to perform PCA, by default False
    pca_components : int, optional
        The number of components to use for PCA, by default max number of components
    plot_scree_plot : bool, optional
        Whether or not to plot the scree plot, by default False
    
    Returns
    -------
    countries : pd.Series
        The countries of the data
    data : pd.DataFrame
        The pre-processed data
    """
    countries = data.pop('country')
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    elif scaler == 'normalizer':
        scaler = Normalizer()
    elif scaler == 'quantile':
        scaler = QuantileTransformer()
    elif scaler == 'power':
        scaler = PowerTransformer()
    elif scaler == 'maxabs':
        scaler = MaxAbsScaler()
    elif scaler == 'function':
        scaler = FunctionTransformer()
    data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    if pca:
        pca = PCA(pca_components)
        data = pca.fit_transform(data)
        data = pd.DataFrame(data, columns=[f'PC{i}' for i in range(1, pca_components+1)])
    if plot_scree_plot and pca:
        scree = list(pca.explained_variance_ratio_*100) # get variance ratios
        labels = ['PC' + str(x) for x in range(1, len(scree)+1)] # make labels for scree plot
        labels = [scree[i] for i in range(len(scree))]
        for i in range(1, len(scree)):
            labels[i] = labels[i] + labels[i-1]
        labels = [round(i, 2) for i in labels]

        # plot the percentage of explained variance by principal component
        plt.bar(x=range(1,len(scree)+1), height=scree, tick_label = labels) 
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component aggregated variance')
        plt.title(f'PCA Scree Plot using {str(scaler)[:-2]}')
        plt.show()
    return countries, data



def create_map_plot(data: pd.DataFrame, output_dir: str):
    """
    data DataFrame needs following columns:
        cluster: id of cluster for each row
        name: country
    """

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    country_geopandas = country_geopandas.merge(
        data, # this should be the pandas with statistics at country level
        how='inner', 
        left_on=['name'], 
        right_on=['name']
    )

    urban_area_map = folium.Map()
    choropleth = folium.Choropleth(
        geo_data=country_geopandas,
        name='choropleth',
        data=country_geopandas,
        columns=['name', 'cluster'],
        key_on='feature.properties.name',
        fill_color='Set1',
        nan_fill_color='Grey',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Cluster ids'
    ).add_to(urban_area_map)
    for key in choropleth._children:
        if key.startswith('color_map'):
            del(choropleth._children[key])
            
    choropleth.add_to(urban_area_map)

    urban_area_map.save(f'{output_dir}/graph_{datetime.now().strftime("%Y-%m-%d-time-%H-%M-%S")}.html')


def create_dendrogram(data: pd.DataFrame):
    return sch.dendrogram(sch.linkage(data, method = 'ward'))


def apply_hierarchical_clustering(data: pd.DataFrame, cluster_num: int = 5):
    """Appends output of hierarchical clustering to provided data in 'cluster' column"""
    agg_hc = AgglomerativeClustering(n_clusters=cluster_num, affinity='euclidean', linkage='ward')
    y_hc = agg_hc.fit_predict(data)
    data["cluster"] = y_hc

    return data