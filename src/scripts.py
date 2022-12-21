import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import geopandas
import folium
from datetime import datetime
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def dist(p1: list, p2: list) -> float:
    """Calculates the Euclidean distance between two points in n-dimensional space (n = len(p1) = len(p2)).
    

    Parameters
    ----------
    p1 : list
        The first point
    p2 : list
        The second point
    
    Returns
    -------
    float
        The Euclidean distance between the two points"""
    return np.sqrt(np.sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))


def calculate_centroid(cluster_df: pd.DataFrame) -> list:
    """Return the centroid of a cluster as a list of the mean of each feature. 
    
    Parameters
    ----------
    cluster_df : pd.DataFrame
        The cluster to calculate the centroid of
    
    Returns
    -------
    list
        The centroid of the cluster"""
    return cluster_df.describe().loc['mean']


def ICV(cluster: pd.DataFrame) -> float:
    """Calculate the Intra-Cluster-Variance (ICV) of the provided cluster.
    This is calculated as the mean of the distances of each data point in a cluster,
    to every other data point in the same cluster.
    
    Parameters
    ----------
    cluster : pd.DataFrame
        The cluster to calculate the ICV of
        
    Returns
    -------
    float
        The ICV of the cluster"""
    
    average_distances = []

    for sample in cluster.values.tolist():
        current = sample

        distances_from_current = []
        for point in cluster.values.tolist():
            if current != point:
                distances_from_current.append(dist(current, point))
        average_distances.append(np.mean(distances_from_current))

    return average_distances


def split_in_clusters(cluster_df: pd.DataFrame) -> dict:
    """Returns a dict with the clusters as values and the cluster number as key
    
    Parameters
    ----------
    cluster_df : pd.DataFrame
        The dataframe with the clusters
        
    Returns
    -------
    result : dict
        The clusters as values and the cluster number as key"""
    result = {} 
    for i in range(len(cluster_df.cluster.unique())):
        result[i] = cluster_df.loc[cluster_df['cluster'] == i]\
            .drop(columns=['cluster'], axis=1)
    return result


def evalutate_clusters(clustered_df: pd.DataFrame):
    """Calculate the silhouette score, Calinski-Harabasz Index, Davies-Bouldin Index of a clustered dataframe (in that order).
    The dataframe needs to have a 'cluster' column, and the rest of the columns are the features.
    
    Parameters
    ----------
    clustered_df : pd.DataFrame
        The dataframe with the clusters
    
    Returns
    -------
    s : float
        The silhouette score
    c : float
        The Calinski-Harabasz Index
    d : float
        The Davies-Bouldin Index
    """
    s = silhouette_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    c = calinski_harabasz_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    d = davies_bouldin_score(clustered_df.drop(columns=['cluster'], axis=1), clustered_df['cluster'])
    The dataframe needs to have a 'cluster' column, and the rest of the columns are the features."""
    s = silhouette_score(
        clustered_df.drop(
            columns=['cluster'],
            axis=1),
        clustered_df['cluster'])
    c = calinski_harabasz_score(
        clustered_df.drop(
            columns=['cluster'],
            axis=1),
        clustered_df['cluster'])
    d = davies_bouldin_score(
        clustered_df.drop(
            columns=['cluster'],
            axis=1),
        clustered_df['cluster'])
    return s, c, d


def display_clusters(df):
def display_clusters(data : pd.DataFrame) -> pd.DataFrame:
    """Display the clusters in a pivot table.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with the clusters

    Returns
    -------
    pd.DataFrame
        The pivot table"""
    return data.groupby(data['cluster']).mean()


def pre_process_data(
        data: pd.DataFrame,
        scaler: str = 'standard',
        pca=False,
        pca_components: int = 9,
        plot_scree_plot: bool = False):
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
        data = pd.DataFrame(
            data, columns=[
                f'PC{i}' for i in range(
                    1, pca_components + 1)])
    if plot_scree_plot and pca:
        scree = list(
            pca.explained_variance_ratio_ *
            100)  # get variance ratios
        labels = [
            'PC' +
            str(x) for x in range(
                1,
                len(scree) +
                1)]  # make labels for scree plot
        labels = [scree[i] for i in range(len(scree))]
        for i in range(1, len(scree)):
            labels[i] = labels[i] + labels[i - 1]
        labels = [round(i, 2) for i in labels]

        # plot the percentage of explained variance by principal component
        plt.bar(x=range(1, len(scree) + 1), height=scree, tick_label=labels)
        plt.ylabel('Percentage of Explained Variance')
        plt.xlabel('Principal Component aggregated variance')
        plt.title(f'PCA Scree Plot using {str(scaler)[:-2]}')
        plt.show()
    return countries, data



def create_map_plot(data: pd.DataFrame, output_dir: str) -> None:
    """
    Create a map plot of the clusters. 
    The data should be a pandas dataframe with the country names as index and the cluster as column.

    Parameters
    ----------
    data : pd.DataFrame
        The data to be plotted
    output_dir : str
        The directory to save the plot to
    
    Returns
    -------
    None
    """

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    )
    country_geopandas = country_geopandas.merge(
        data,  # this should be the pandas with statistics at country level
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
            del (choropleth._children[key])

    choropleth.add_to(urban_area_map)

    urban_area_map.save(
        f'{output_dir}/graph_{datetime.now().strftime("%Y-%m-%d-time-%H-%M-%S")}.html')

def create_dendrogram(data: pd.DataFrame) -> sch.dendrogram:
    """Creates a dendrogram of the hierarchical clustering
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to be clustered
    
    Returns
    -------
    dendrogram
        The dendrogram"""
    return sch.dendrogram(sch.linkage(data, method = 'ward'))


def apply_hierarchical_clustering(data: pd.DataFrame, cluster_num: int = 5) -> pd.DataFrame:
    """Appends output of hierarchical clustering to provided data in 'cluster' column
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to be clustered
    cluster_num : int, optional
        The number of clusters to be used, by default 5
        
    Returns
    -------
    pd.DataFrame
        The data with the cluster column appended"""
    agg_hc = AgglomerativeClustering(n_clusters=cluster_num, affinity='euclidean', linkage='ward')
    y_hc = agg_hc.fit_predict(data)
    data["cluster"] = y_hc

    return data


def wcss(cluster: pd.DataFrame) -> float:
    """Calculate the Within-Cluster-Sum-of-Squares (WCSS) for a given cluster"""
    cluster = cluster.values.tolist()

    # Calculate the centroid of the cluster
    centroid = np.mean(cluster, axis=0)

    # Initialize the WCSS to 0
    wcss = 0

    # Iterate over each data point in the cluster
    for point in cluster:
        # Calculate the squared distance between the data point and the
        # centroid
        squared_distance = np.sum((point - centroid)**2)
        # Add the squared distance to the WCSS
        wcss += squared_distance

    # WCSS is the sum of the squared distances for all the data points in the
    # cluster
    return wcss


def gap_statistic(df: pd.DataFrame, n_clusters: int, plot_gap: bool = True) -> float:
    """Iteratively calculate and plot the gap statistic for a given dataset and number of clusters provided.
      Pandas dataframe provided most only contain numerical values."""

    df = np.array(df.values.tolist())  # Convert the dataframe to a numpy array
    gaps = []
    ks = np.arange(2, n_clusters)

    for k in ks:
        # Use KMeans to cluster the data into n_clusters clusters
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)

        # Calculate the WCSS of the clusters
        wcss = kmeans.inertia_

        # Generate a reference distribution of the data by randomly assigning
        # the data points to clusters
        reference_distribution = np.random.randint(
            low=0, high=k - 1, size=df.shape[0])

        # Calculate the WCSS for the reference distribution
        reference_wcss = 0
        for i in range(k):
            cluster = df[reference_distribution == i]
            centroid = np.mean(cluster, axis=0)
            for point in cluster:
                squared_distance = np.sum((point - centroid)**2)
                reference_wcss += squared_distance

        # Calculate the gap statistic as the difference between the WCSS of the
        # clusters and the WCSS of the reference distribution, normalized by
        # the WCSS of the reference distribution
        gap = (wcss - reference_wcss) / reference_wcss
        gaps.append(gap)

    if plot_gap:
        plt.plot(ks, gaps)
        plt.grid()
        plt.ylabel("Gap Statistic")
        plt.xlabel("Number of Clusters, k")

    return gaps
