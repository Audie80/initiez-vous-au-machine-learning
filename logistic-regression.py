import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
# split train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)

# initialisation et entraînement en une ligne
clf = LogisticRegression(random_state=8).fit(X_train, y_train)

# prediction de la catégorie de chaque échantillon sur le set de test
y_pred = clf.predict(X_test)
print(y_pred)

# prédiction de la probabilité d'appartenance à chaque catégorie pour chaque échantillon du set de test
y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba)

# histogramme des probabilités des prédictions pour analyser les performances du modèle de classification
y_hat_proba = clf.predict_proba(X)[:, 1]
sns.histplot(y_hat_proba)
plt.savefig("logistic-regression-histogram.png")

# exactitude = échantillons bien classés / échantillons au total
y_pred_2 = clf.predict(X)
print("Accuracy score: ", accuracy_score(y, y_pred_2))

# matrice de confusion = tableau croisé des prédictions vs la réalité
print("Confusion matrix: ", confusion_matrix(y, y_pred_2))

# Receiver Operating Characteristic (ou ROC), fonction d’efficacité du récepteur
print("ROC-AUC", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))
