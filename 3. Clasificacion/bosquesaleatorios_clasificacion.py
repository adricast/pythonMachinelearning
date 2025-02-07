#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:55:27 2019

@author: juangabriel
"""

# Random Forest Classification

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el data set
dataset = pd.read_csv('Social_Network_Ads.csv')

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Crea un DataFrame para las dos columnas de X
x_df = pd.DataFrame(X, columns=['X1', 'X2'])

# Agrega la columna y al DataFrame
comparison_df = x_df.copy()
comparison_df['y'] = y

# Muestra la tabla resultante
print(comparison_df)

# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# Ajustar el clasificador  Random Forest en el Conjunto de Entrenamiento
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)

# Predicción de los resultados con el Conjunto de Testing
y_pred  = classifier.predict(X_test)

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#   63                  |  5   falso Positivo
#   10  falso negativo  |  80 

print(cm)
# evaluar falsos positivos y negativos
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)
tp, fp = cm[0, 0], cm[0, 1]
fn, tn = cm[1, 0], cm[1, 1]

# Cálculos
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
fpr = fp / (fp + tn)
fnr = fn / (fn + tp)

print(f"Precisión (Accuracy): {accuracy:.2%}")
print(f"Tasa de Error: {error_rate:.2%}")
print(f"Tasa de Falsos Positivos (FPR): {fpr:.2%}")
print(f"Tasa de Falsos Negativos (FNR): {fnr:.2%}")


# Representación gráfica de los resultados del algoritmo en el Conjunto de Entrenamiento
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest  (Conjunto de Entrenamiento)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()


# Representación gráfica de los resultados del algoritmo en el Conjunto de Testing
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Random Forest (Conjunto de Test)')
plt.xlabel('Edad')
plt.ylabel('Sueldo Estimado')
plt.legend()
plt.show()

#CURVAS DE CAP
import numpy as np
import matplotlib.pyplot as plt

# Obtener probabilidades de predicción en lugar de clases directamente
y_probs = classifier.predict_proba(X_test)[:, 1]  # Probabilidad de la clase positiva

# Combinar las probabilidades con los valores reales
data = list(zip(y_probs, y_test))
data.sort(reverse=True, key=lambda x: x[0])  # Ordenar por probabilidad descendente

# Contadores para valores positivos y totales
total_positives = sum(y_test)
total_negatives = len(y_test) - total_positives

# Variables para las curvas
x_points = [0]
y_points = [0]
positives_accumulated = 0

for i, (_, actual) in enumerate(data):
    if actual == 1:
        positives_accumulated += 1
    x_points.append((i + 1) / len(y_test))
    y_points.append(positives_accumulated / total_positives)

# Línea ideal (clasificación perfecta)
ideal_x = [0, total_positives / len(y_test), 1]
ideal_y = [0, 1, 1]

# Línea aleatoria (baseline)
random_x = [0, 1]
random_y = [0, 1]

# Plot de la curva CAP
plt.figure(figsize=(10, 6))
plt.plot(x_points, y_points, label='Curva CAP', color='blue')
plt.plot(ideal_x, ideal_y, label='Curva Ideal', linestyle='--', color='green')
plt.plot(random_x, random_y, label='Clasificación Aleatoria', linestyle='--', color='red')
plt.title('Curva CAP - Random Forest')
plt.xlabel('Porcentaje acumulativo de observaciones')
plt.ylabel('Porcentaje acumulativo de positivos')
plt.legend()
plt.show()
