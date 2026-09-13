from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# split train, test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)

# initialisation et entraînement en une ligne
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=8).fit(X_train, y_train)

# prediction de la catégorie de chaque échantillon sur le set de test
y_pred = clf.predict(X_test)
print(y_pred)

# prédiction de la probabilité d'appartenance à chaque catégorie pour chaque échantillon du set de test
y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba)

# histogramme des probabilités des prédictions pour analyser les performances du modèle de classification
y_hat_proba = clf.predict_proba(X)[:,1]
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(y_hat_proba)
plt.savefig("logistic-regression-histogram.png")

# exactitude = échantillons bien classés / échantillons au total
y_pred_2 = clf.predict(X)
from sklearn.metrics import accuracy_score
print("Accuracy score: ", accuracy_score(y, y_pred_2))

# matrice de confusion = tableau croisé des prédictions vs la réalité
from sklearn.metrics import confusion_matrix
print("Confusion matrix: ", confusion_matrix(y, y_pred_2))

# Receiver Operating Characteristic (ou ROC), fonction d’efficacité du récepteur
from sklearn.metrics import roc_auc_score
print("ROC-AUC", roc_auc_score(y_test, clf.predict_proba(X_test),multi_class='ovr'))
