import pandas as pd
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import numpy as np

def predire_age(chemin_fichier_json, choix_modele):
    # Charger les données de test à partir du fichier JSON
    with open(chemin_fichier_json, 'r', encoding='utf-8') as fichier:
        donnees = json.load(fichier)

    # Créer un DataFrame Pandas à partir des données JSON
    donnees_test = pd.DataFrame(donnees)

    # Charger les modèles
    fichiers_modeles = {
        'knn': 'best_knn_model.pkl',
        'arbre_decision': 'best_decision_model.pkl',
        'foret_aleatoire': 'best_rf_model.pkl',
        'gbr': 'best_GBR_model.pkl'
    }

    # Charger le modèle en fonction du choix de l'utilisateur
    if choix_modele in fichiers_modeles:
        with open(fichiers_modeles[choix_modele], 'rb') as fichier:
            modele = pickle.load(fichier)
    else:
        raise ValueError("Modèle non supporté.")

    # Supposer que x_test est préparé à partir de donnees_test
    # Remplacer par la logique réelle d'extraction des caractéristiques
    # Supposons que 'age' est la colonne cible

    # Faire la prédiction
    predictions = modele.predict(x_test)
    predictions = pd.DataFrame(predictions)

    # Sauvegarder les résultats sous forme de JSON                          
    fichier_resultat = 'prediction.json'
    predictions.to_json(fichier_resultat, orient='records', lines=True)

    return predictions.to_json(orient='records', lines=True)


if __name__ == "__main__":

# Appel de la fonction pour obtenir les prédictions au format JSON
    chemin_fichier_json = 'test_data.json'
    choix_modele = 'knn'  # Ou 'arbre_decision', 'foret_aleatoire', 'gbr', ''
    ages_predits_json = predire_age(chemin_fichier_json, choix_modele)
    print(ages_predits_json)