# entraînement du modèle de régression linéaire sur le dataset age_vs_taille_vs_poids_vs_sexe.csv
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/age_vs_taille_vs_poids_vs_sexe.csv')

# les variables prédictives
X = df[['sexe', 'age', 'taille']]

# la variable cible, le poids
y = df.poids

# on choisit un modèle de régression linéaire
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X, y)

# et on obtient directement un score
print(reg.score(X, y))

# ainsi que les coefficients a,b,c de la régression linéaire
print(reg.coef_)
