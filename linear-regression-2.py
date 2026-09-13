# prédiction du poids à partir de l'âge, de la taille et du sexe avec un modèle de régression linéaire
import pandas as pd
df = pd.read_csv('data/age_vs_taille_vs_poids_vs_sexe.csv')

# les variables prédictives
X = df[['sexe','age', 'taille']]

# la variable cible, le poids
y = df.poids

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X, y)

# Exemple de donnée à prédire
import numpy as np
nouvelle_donnee = pd.DataFrame(
    [[0, 150, 153]],
    columns=['sexe', 'age', 'taille']
)

prediction = reg.predict(nouvelle_donnee)
print(prediction)
