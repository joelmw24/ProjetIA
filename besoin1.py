import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def nbr_cluster(n): 
    data = pd.read_csv("Data_Arbre.csv")
    colomns = ["haut_tot", "longitude", "latitude"]
    reduit = data[colomns].dropna()
    X = reduit[['haut_tot']].values
    nb_clusters = n 
    kmeans = KMeans (n_clusters=nb_clusters, random_state=0).fit(X)
    silhouette = silhouette_score(X, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(X,kmeans.labels_)
    calinski = calinski_harabasz_score(X, kmeans.labels_)
    print('Silhouette score : ', silhouette)
    print('Score Davies Bouldin: ', davies_bouldin)
    print('Score de Calinski Harabasz : ', calinski)

   
print('Choississez le nombre de catégorie que vous souhaitez --> 2 = petit et grand ou 3 =petit moyen grand ', n )

nbr_cluster(n)