# Importer les bibliothèques nécessaires
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Charger les données
data = pd.read_csv('donnees_risque_maladie.csv')

# Séparer les caractéristiques et la cible
X = data.drop('target', axis=1)
y = data['target']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer le modèle
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

# Évaluer le modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# # Prédiction pour un nouveau cas (exemple)
# new_data = [/* valeurs des caractéristiques */]
# new_data_scaled = scaler.transform([new_data])
# prediction = model.predict(new_data_scaled)
# print(f"Prediction: {'Maladie présente' if prediction[0][0] > 0.5 else 'Maladie absente'}")
