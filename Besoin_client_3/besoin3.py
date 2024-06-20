import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def load_model(model_file):
    with open(model_file, 'rb') as file:
        best_sgd_model = pickle.load(file)
    return best_sgd_model

def load_test_data(data_json_file):
    with open(data_json_file, 'r') as file:
        data_json = json.load(file)
    data_map = pd.DataFrame(data_json)
    data_map = data_map.drop('fk_arb_etat', axis=1)
    return data_map

def predict_data(model, data):
    predictions = model.predict(data)
    data['predictions'] = predictions
    return data, predictions

def plot(data, predictions):
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(data['longitude'], data['latitude'], c=predictions, cmap='viridis', marker='x', label='Arbres à risque')
    plt.colorbar(sc, label='Risque de déracinement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Carte des arbres en risque de déracinement')
    plt.grid(True)
    plt.legend()
    plt.show()

def map(data, predictions):
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

def heatmap(data):

    plt.figure(figsize=(10, 8))
    sns.kdeplot(data=data, x='longitude', y='latitude', fill=True, cmap='Reds')
    plt.title('Heatmap des arbres en risque de déracinement')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()

if __name__ == "__main__":
    model_file = 'best_gbm_model.pkl'
    best_sgd_model = load_model(model_file)

    data_json_file = 'test_data.json'
    data_map = load_test_data(data_json_file)
    data_map, predictions = predict_data(best_sgd_model, data_map) # Prédictions sur les données

    plot(data_map, predictions)    # grid
    map(data_map, predictions)     # carte interactive 
    heatmap(data_map)              # heatmap 
