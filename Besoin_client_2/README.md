# Client 2
Le but de cette partie est de predire l'age des des arbres à partir de différents modeles de regressions.

## La base de données 
La base de données qu'on utilise est Data_Arbre.csv.

## Fichier Notebook 
Le fichier besoin2.ipynb est le fichier Notebook où l'on retrouve les différents modèles testés

## Fichier python 
Le fichier besoin2.py est le script python.
Une fois ce fichier pyhton lancé, mettez le nom du modele que vous allez tester.


# Appel de la fonction pour obtenir les prédictions au format JSON
    chemin_fichier_json = 'test_data.json'
    choix_modele = 'knn'  # Ou 'arbre_decision', 'foret_aleatoire', 'gbr', ''
    ages_predits_json = predire_age(chemin_fichier_json, choix_modele)
    print(ages_predits_json)