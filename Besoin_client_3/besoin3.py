import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

def load_and_predict(model_file, data_json_file):
    # Charger le modèle depuis le fichier
    with open(model_file, 'rb') as file:
        best_model = pickle.load(file)
    
    # Charger les données de test depuis le fichier JSON
    with open(data_json_file, 'r') as file:
        data_json = json.load(file)
    data_map = pd.DataFrame(data_json)
    data_map = data_map.drop('fk_arb_etat', axis=1, errors='ignore')
    
    # Effectuer les prédictions
    predictions = best_model.predict(data_map)
    data_map['predictions'] = predictions
    
    return data_map, predictions

def plot_grid(data):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(data['longitude'], data['latitude'], c=data['predictions'], cmap='viridis', marker='x', label='Arbres à risque')
    plt.colorbar(sc, label='Risque de déracinement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Carte des arbres en risque de déracinement')
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_map(data):
    fig = px.scatter_mapbox(
        data,
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

def plot_heatmap(data):
    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=data, x='longitude', y='latitude', fill=True, cmap='Reds')
    plt.title('Heatmap des arbres en risque de déracinement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

if __name__ == "__main__":
    model_file = 'best_gbm_model.pkl' 
    data_json_file = 'test_data.json' 
    
    data_map, predictions = load_and_predict(model_file, data_json_file)

    plot_grid(data_map)    # Carte en grille
    plot_map(data_map)     # Carte interactive 
    plot_heatmap(data_map) # Heatmap
