#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Apr 2 19:05:55 2019
@author: juangabriel
"""

# Importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyECLAT import ECLAT  # Importación correcta

# Importar el dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Preprocesar los datos: Convertir las transacciones en un DataFrame binario (One-Hot Encoding)
transactions = []
for i in range(len(dataset)):  # Usar len(dataset) para mayor flexibilidad
    transactions.append([str(dataset.values[i, j]) for j in range(dataset.shape[1]) if str(dataset.values[i, j]) != 'nan'])

# Crear el DataFrame de pandas
df_transactions = pd.DataFrame(transactions)

# Crear el objeto ECLAT
eclat_instance = ECLAT(df_transactions)

# Obtener los itemsets frecuentes con un soporte mínimo de 0.003
frequent_itemsets, supports = eclat_instance.fit(min_support=0.003, min_combination=1, max_combination=3)

# Función para imprimir los resultados
def print_results(itemsets, supports):
    print("\nRESULTADOS DE LOS ITEMSETS FRECUENTES (ECLAT):")
    for idx, (itemset, support) in enumerate(zip(itemsets.keys(), supports.values())):
        print(f"\nItemset {idx + 1}: {set(itemset)}")
        print(f"Soporte: {support:.4f}")

# Imprimir los itemsets frecuentes
print_results(frequent_itemsets, supports)

# Visualización: Mostrar los itemsets con mayor soporte
sorted_items = sorted(supports.items(), key=lambda x: x[1], reverse=True)[:5]

print("\nTop 5 itemsets con mayor Soporte:")
for i, (itemset, support) in enumerate(sorted_items):
    print(f"\nItemset {i + 1}: {set(itemset)}")
    print(f"Soporte: {support:.4f}")

# Gráfica para visualizar el soporte de los itemsets
plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_items)), [support for _, support in sorted_items], color='skyblue')
plt.yticks(range(len(sorted_items)), [str(set(itemset)) for itemset, _ in sorted_items])
plt.xlabel("Soporte")
plt.title("Soporte de los itemsets frecuentes (ECLAT)")
plt.gca().invert_yaxis()  # Invertir el eje Y para que los más frecuentes estén arriba
plt.show()
