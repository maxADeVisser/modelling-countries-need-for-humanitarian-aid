import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import geopandas
import folium
from datetime import datetime
from sklearn.cluster import KMeans
import io
from PIL import Image
import branca.colormap as cm

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
    
    #return cluster_centroid_distances # np.sum(intra_variances)


def pre_process_data(data: pd.DataFrame):
    """Make into a function that can be imported and perform all pre-processing steps"""
    # TODO make this
    
    # scaling
    
    pass


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

    #img_data = urban_area_map._to_png(5)
    #img = Image.open(io.BytesIO(img_data))
    #img.save(f'{output_dir}/graph_{datetime.now().strftime("%Y-%m-%d-time-%H-%M-%S")}.png')
