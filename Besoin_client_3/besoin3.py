import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

model_file = 'best_sgd_model.pkl'
with open(model_file, 'rb') as file:
    best_model = pickle.load(file)

csv_file = 'Data_Arbre.csv'
json_file = 'test_data.json'
data = pd.read_csv(csv_file)
columns = ['longitude', 'latitude', 'haut_tot', 'haut_tronc', 'tronc_diam', 'age_estim', 'fk_arb_etat']
data = data[columns]
data.to_json(json_file, orient='records')

# Charger les données de test depuis le fichier JSON
with open(json_file, 'r') as file:
    data_json = json.load(file)

data_map = pd.DataFrame(data_json)
data_map = data_map.drop('fk_arb_etat', axis=1)

# Effectuer les prédictions avec le modèle
predictions = best_model.predict(data_map)
data_map['predictions'] = predictions

# Afficher la carte en grille avec Matplotlib
plt.figure(figsize=(10, 8))
sc = plt.scatter(data_map['longitude'], data_map['latitude'], c=data_map['predictions'], cmap='viridis', marker='o', label='Arbres à risque')
plt.colorbar(sc, label='Risque de déracinement')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Carte des arbres en risque de déracinement')
plt.grid(True)
plt.legend()
plt.show()

fig = px.scatter_mapbox(
    data_map,
    lat='latitude',
    lon='longitude',
    color='predictions',
    title='Carte des arbres en risque de déracinement à Saint-Quentin'
)

fig.update_layout(
    mapbox=dict(
        center=dict(lat=49.8489, lon=3.2876),
        zoom=13
    ),
    mapbox_style='open-street-map',
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

fig.show()

# Afficher la heatmap avec Seaborn
plt.figure(figsize=(10, 8))
sns.kdeplot(data=data_map, x='longitude', y='latitude', fill=True, cmap='Reds')
plt.title('Heatmap des arbres en risque de déracinement')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()
