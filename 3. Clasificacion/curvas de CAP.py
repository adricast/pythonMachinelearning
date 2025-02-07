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
