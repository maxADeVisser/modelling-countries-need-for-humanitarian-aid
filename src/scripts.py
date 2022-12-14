import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import geopandas
import folium
from datetime import datetime
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, MaxAbsScaler, FunctionTransformer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.ticker as mtick
from IPython.display import display


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

def scatter_plot_data(df: pd.DataFrame, columns: list[str], hover_name: list[str], three: bool = False):
    """Plot the data in a scatter plot. If three is True, a 3D scatter plot is plotted.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot
    columns : list[str]
        The columns to plot
    hover_name : list[str]
        The columns to show in the hover
    three : bool, optional
        Whether to plot a 3D scatter plot, by default False
        
    Returns
    -------
    None
        The plot is shown"""
    if three:
        fig = px.scatter_3d(df, x=columns[0], y=columns[1], z=columns[2], color='cluster', hover_name=hover_name)
        fig.show()
    else:
        fig = px.scatter(df, 
                         x=columns[0], 
                         y=columns[1], 
                         color='cluster', 
                         hover_name=hover_name,
                         width=600,
                         height=500)
        fig.show()
    

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

def dbscan_epsilon(df: pd.DataFrame, min_samples: int):

    k = min_samples if min_samples > 2 else 2 # use min_samples as a heuristic for k

    nbrs = NearestNeighbors(n_neighbors=k).fit(df) # instanciate NearestNeighbors object
    distances, indices = nbrs.kneighbors(df)
    #distances = np.sort(distances, axis=0)[:,1] # sort the distances and keep only the second column
    furthest = np.sort(distances[:, -1])

    plt.figure(figsize=(5,5))
    plt.plot(furthest)
    plt.xlabel('Data points', fontsize=10)
    plt.ylabel(f'Distance to the {k}th-nearest neighbor', fontsize=10)
    plt.grid()

def cal_silhouette(clustered_df: pd.DataFrame):
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
    s = silhouette_score(
        clustered_df.drop(
            columns=['cluster'],
            axis=1),
        clustered_df['cluster'])
    # c = calinski_harabasz_score(
    #     clustered_df.drop(
    #         columns=['cluster'],
    #         axis=1),
    #     clustered_df['cluster'])
    # d = davies_bouldin_score(
    #     clustered_df.drop(
    #         columns=['cluster'],
    #         axis=1),
    #     clustered_df['cluster'])
    return s


def pivot_table(clusterd_df : pd.DataFrame, raw_df:pd.DataFrame) -> pd.DataFrame:
    """Display the clusters in a pivot table. OBS. clustered_df needs to originate from raw_df.

    Parameters
    ----------
    clusterd_df : pd.DataFrame
        The dataframe with the clusters
    
    data : pd.DataFrame
        The dataframe with the clusters

    Returns
    -------
    pd.DataFrame
        The pivot table of the raw_df with the clusters"""
    raw_df['cluster'] = clusterd_df['cluster'] 
    pivot_table = raw_df.groupby(raw_df['cluster']).mean()
    return raw_df, pivot_table


def pre_process_data(
        data: pd.DataFrame,
        scaler = None,
        pca=False,
        pca_components = 9,
        plot_scree_plot: bool = False,
        biplot: bool = False,
        plot_pc: tuple[str, str] = None,
        ):
    """Make into a function that can be imported and perform all pre-processing steps
    on the data. This includes scaling, PCA, etc.


    Parameters
    ----------
    data : pd.DataFrame
        The data to be pre-processed
    scaler : str, optional
        The scaler to use, by default None.
        Alternatives: {'standard','minmax', 'robust', 'normalizer', 'quantile', 'power', 'maxabs', 'function'}
    pca : bool, optional
        Whether or not to perform PCA, by default False
    pca_components : int, optional
        The number of components to use for PCA, by default 9
    plot_scree_plot : bool, optional
        Whether or not to plot the scree plot, by default False
    biplot : bool, optional
        Whether or not to plot the biplot, by default False
    Returns
    -------
    countries : pd.Series
        The countries of the data
    data : pd.DataFrame
        The pre-processed data
    pca_components : pd.DataFrame
        The PCA components if pca is True
    """
    
    countr = 0
    if 'country' in data.columns:
        countr = 1
        countries = data['country']
        data = data.drop(columns=['country'], axis=1)
    
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
    if scaler != None:    
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    if pca:
        features = data.columns
        pca = PCA(pca_components)
        pca_data = pca.fit_transform(data)
        data = pd.DataFrame(pca_data, 
                            columns=[f'PC{i}' for i in range(1, pca_components + 1)])
        
        pca_explained = pd.DataFrame(pca.components_, index=data.columns, columns=features)
        
    if (plot_scree_plot and pca):
        explained_variance = list(pca.explained_variance_ratio_ * 100)  # get variance ratios, y
        percentages = [round(i, 2) for i in explained_variance] 
        labels = ['PC' + str(x) for x in range(1, len(explained_variance)+1)] # x axis labels
        
        # plot the percentage of explained variance by principal component
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(x=labels, height=percentages, tick_label=labels)
        ax.bar_label(bars)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        
        plt.ylabel('% of explained variance')
        plt.xlabel('Principal Components')
        plt.show()
    
    if (biplot and pca):
        # source https://stackoverflow.com/questions/50796024/feature-variable-importance-after-a-pca-analysis
        fig, ax = plt.subplots(figsize=(7,6), dpi=100)
        ax.scatter(data['PC1'], data['PC2'], alpha=0.4)
        
        for i in pca_explained:
            current_arrow = pca_explained[i][:2]
            ax.arrow(0, 
                     0, 
                     current_arrow[0], 
                     current_arrow[1], 
                     color = 'r', 
                     alpha = 0.5, 
                     length_includes_head=True,
                     head_width=0.02,
                     head_length=0.02)
            ax.text(current_arrow[0]*1.15, current_arrow[1]*1.15, i, color="g", ha = 'center', va = 'center', fontsize=8)
        plt.grid()
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        fig.show()
        
    if (countr == 1 and pca):
        return countries, data, pca_explained
    elif countr == 1:
        return countries, data
    else:
        return data

def create_map_plot(data: pd.DataFrame, out_dir: str = None) -> None:
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
    
    country_rename = {
        "Central African Republic" : "Central African Rep.",
        "Congo, Dem. Rep.": "Dem. Rep. Congo",
        "Congo, Rep." : "Congo",
        "Equatorial Guinea": "Eq. Guinea",
        "Macedonia, FYR": "North Macedonia",
        "Bosnia and Herzegovina":"Bosnia and Herz.",
        "Slovak Republic":"Slovakia",
        "Czech Republic":"Czechia",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Cote d'Ivoire":"C??te d'Ivoire",
        "United States":"United States of America",
        "Dominican Republic": "Dominican Rep.",
        "Lao": "Laos",
    }
    data['country'] = data['country'].replace(country_rename)

    country_geopandas = geopandas.read_file(
        geopandas.datasets.get_path('naturalearth_lowres')
    ).rename(columns={'name': 'country'})
    country_geopandas = country_geopandas.merge(
        data,  # this should be the pandas with statistics at country level
        how='inner',
        left_on=['country'],
        right_on=['country']
    )

    urban_area_map = folium.Map()
    choropleth = folium.Choropleth(
        geo_data=country_geopandas,
        name='choropleth',
        data=country_geopandas,
        columns=['country', 'cluster'],
        key_on='feature.properties.country',
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
    if out_dir:
        urban_area_map.save(f'{out_dir}/graph_{datetime.now().strftime("%Y-%m-%d-time-%H-%M-%S")}.html')
    else:
        return urban_area_map
    

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
    cluster = cluster.values.tolist() # convert pd to numpy array

    # Calculate the centroid of the cluster
    centroid = np.mean(cluster, axis=0)

    # Initialize the WCSS to 0
    wcss = 0

    # Iterate over each data point in the cluster
    for point in cluster:
        # Calculate the squared distance between the data point and the centroid
        squared_distance = np.sum((point - centroid)**2)
        # Add the squared distance to the WCSS
        wcss += squared_distance

    return wcss

def apply_dbscan(df: pd.DataFrame(), eps, min_samples) -> pd.DataFrame:
    """Apply DBSCAN to the provided data and return the components and the data with the cluster column appended"""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(df.drop("country", axis=1))
    
    clusters = dbscan.labels_
    df["cluster"] = clusters
    return dbscan.components_, df


def gap_statistic(df: pd.DataFrame, n_clusters: int, plot_gap: bool = True) -> float:
    """Iteratively calculate and plot the gap statistic for a given dataset and number of clusters provided.
        Pandas dataframe provided most only contain numerical values.     

    Parameters
    ----------
    df : pd.DataFrame
        The data to be clustered
    n_clusters : int
        The number of clusters to be used
    plot_gap : bool, optional
        Whether to plot the gap statistic, by default True
    
    Returns
    -------
    float
        The optimal number of clusters"""

    df = np.array(df.values.tolist())  # Convert the dataframe to a numpy array
    gaps = []
    ks = np.arange(2, n_clusters + 1)

    for k in ks:
        # Use KMeans to cluster the data into n_clusters clusters
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)

        # Calculate the WCSS of the clusters
        wcss = kmeans.inertia_

        # Generate a reference distribution of the data by randomly assigning
        # the data points to clusters
        reference_distribution = np.random.randint(
            low=1, high=k, size=df.shape[0])

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