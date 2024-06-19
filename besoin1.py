import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score


def nbr_cluster(n): 
    data = pd.read_csv("C:\\Users\\anais\\Downloads\\Data_Arbre.csv")
    colomns = ["haut_tot", "longitude", "latitude"]
    reduit = data[colomns].dropna()
    X = reduit[['haut_tot']].values
    nb_clusters = n 
    kmeans = KMeans(n_clusters=nb_clusters, random_state=0).fit(X)
    reduit['cluster'] = kmeans.labels_
    silhouette = silhouette_score(X, kmeans.labels_)
    davies_bouldin = davies_bouldin_score(X,kmeans.labels_)
    calinski = calinski_harabasz_score(X, kmeans.labels_)
    print('Silhouette score : ', silhouette)
    print('Score Davies Bouldin: ', davies_bouldin)
    print('Score de Calinski Harabasz : ', calinski)
    
    fig = px.scatter_mapbox(
        reduit,
        lat = 'latitude',
        lon ='longitude',
        #hover_name = 'haut_tot',
        color = 'cluster',
        zoom = 12,
        height = 600, 
    )

    fig.update_layout(mapbox_style = "open-street-map")
    fig.update_layout(margin = {"r":0,"t":0,"l":0,"b":0})
    fig.show()
    from sklearn.ensemble import IsolationForest

    #Récupérer les données pertinentes
    colomns_anomalies = data[["age_estim", "tronc_diam", "haut_tronc"]]
    #Effectuer un IsolationForest contamnoation -> valeur de base + augmente plus ca detecte une anomalies , choisi tous 42
    isolation = IsolationForest(contamination=0.05, random_state=42)
    reduit['anomaly']= isolation.fit_predict(colomns_anomalies)

    anomalies = reduit[reduit['anomaly'] ==-1]
    nb_anomalies = np.count_nonzero(reduit['anomaly'] == -1)
    print("Nombre d'anomalies détéctées: ", nb_anomalies)

   
n = int(input('Veuillez choisir un nombre de cluster : '))

nbr_cluster(n)