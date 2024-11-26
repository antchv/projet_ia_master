import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# Charger les données (à remplacer par le chemin de votre fichier CSV)
data = pd.read_csv('./donnees_sante.csv')

# Séparer les caractéristiques (features) et la cible (target)
X = data.iloc[:, :-1]  # toutes les colonnes sauf la dernière
y = data.iloc[:, -1]   # la dernière colonne

# Séparer les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normaliser les caractéristiques
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer un modèle de réseau de neurones
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle et stocker l'historique
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test))

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy}')

# Tracer le graphique de la précision
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Modèle Précision')
plt.ylabel('Précision')
plt.xlabel('Epoch')
plt.legend(['Entraînement', 'Test'], loc='upper left')
plt.show()

# Tracer le graphique de la perte
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Modèle Perte')
plt.ylabel('Perte')
plt.xlabel('Epoch')
plt.legend(['Entraînement', 'Test'], loc='upper left')
plt.show()

# Prédiction pour le patient test
patient_data = np.array([[1, 12, 70, 27, 0, 36.8, 0.342, 27]])
patient_data = scaler.transform(patient_data)  # Utiliser le même scaler que pour les données d'entraînement
prediction = model.predict(patient_data)
predicted_class = (prediction > 0.5).astype(int)
print(f'Le modèle prédit que le patient est {"malade" if predicted_class[0][0] == 1 else "non malade"}')