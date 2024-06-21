# Notebook Overview


L'objectif principal de cette analyse est de comprendre l'impact des différentes variables sur le résultat. Nous allons examiner les données, effectuer des analyses statistiques et construire des modèles prédictifs pour tirer des conclusions significatives.

## Préparation des données

```python
import pandas as pd
data = pd.read_csv('data.csv')
data.head()
```

```python
from sklearn.model_selection import train_test_split

# Séparation en ensembles d'entraînement (60%), de validation (25%) et de test (15%)
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=42)
```

## Apprentissage Supervisé pour la classification

```python
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train_scaled, y_train)
y_pred = sgd_clf.predict(x_val_scaled)
```


## Métriques pour la classification

```python
# Validation croisée 
scores_res = cross_val_score(sv_res, x_train_res, y_train_res, cv=3, scoring='accuracy')

# Affichage des résultats
print("Scores de précision pour chaque fold  :", scores_res)
print("Précision moyenne  :", scores_res.mean())
print("Matrice de confusion (test) :\n", confusion_matrix(y_test, y_pred_test, normalize='true'))
print("\nRapport de classification (test) :\n", classification_report(y_test, y_pred_test))
```

```python
# Stochastic Gradient Descent GridSearch
param_grid_sgd = {
    'loss': ['hinge', 'modified_huber', 'squared_hinge'],
    'penalty': ['l2', 'l1', 'elasticnet'],
    'alpha': [0.0001, 0.001, 0.01],
    'max_iter': [1000, 2000, 3000],
}

grid_search_sgd = GridSearchCV(SGDClassifier(random_state=42), param_grid_sgd, cv=3, scoring='accuracy', verbose=2)
grid_search_sgd.fit(x_train_scaled, y_train)  # entraînement 
```