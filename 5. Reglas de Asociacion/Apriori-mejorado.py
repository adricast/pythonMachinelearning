#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
"""
Created on Tue Apr 2 19:05:55 2019

@author: juangabriel
"""

# Apriori

# Cómo importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)

# Preprocesar los datos y preparar las transacciones
transactions = []
for i in range(0, len(dataset)):  # Usar len(dataset) en lugar de 7501 para mayor flexibilidad
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])  # Considerando hasta 20 productos por transacción

# Entrenar el algoritmo de Apriori
from Apyori import apriori
rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

# Convertir las reglas a una lista
results = list(rules)

# Función para imprimir las reglas de manera legible
def print_results(results):
    print("RESULTADOS DE LAS REGLAS DE ASOCIACIÓN:")
    for idx, rule in enumerate(results):
        print(f"\nRegla {idx + 1}:")
        
        # Recorrer las estadísticas ordenadas
        for ordered_stat in rule.ordered_statistics:
            base = ordered_stat.items_base  # Elementos base de la regla
            items_added = ordered_stat.items_add  # Elementos añadidos por la regla
            confidence = ordered_stat.confidence  # Confianza de la regla
            lift = ordered_stat.lift  # Lift de la regla
            
            # Mostrar la base y los items añadidos
            print(f"Base: {base}")
            print(f"Items Añadidos: {items_added}")
            
            # Mostrar soporte, confianza y lift
            print(f"Soporte: {rule.support}")
            print(f"Confianza: {confidence}")
            print(f"Lift: {lift}")

# Imprimir todos los resultados de las reglas
print_results(results)

# Visualización: Mostrar las reglas con mayor lift
# Crear una lista con los lifts de cada regla
lifts = [ordered_stat.lift for rule in results for ordered_stat in rule.ordered_statistics]

# Ordenar las reglas por lift (de mayor a menor)
sorted_lifts = sorted(zip(lifts, results), reverse=True)

# Mostrar las 5 reglas con mayor lift
print("\nTop 5 reglas con mayor Lift:")
for i in range(5):
    lift, rule = sorted_lifts[i]
    print(f"\nRegla {i + 1} con Lift: {lift}")
    for ordered_stat in rule.ordered_statistics:
        print(f"Base: {ordered_stat.items_base}")
        print(f"Items Añadidos: {ordered_stat.items_add}")
        print(f"Confianza: {ordered_stat.confidence}")
        print(f"Lift: {ordered_stat.lift}")

# Gráfica para visualizar el lift de las reglas
plt.figure(figsize=(10, 6))
plt.barh(range(len(lifts)), lifts, color='skyblue')
plt.yticks(range(len(lifts)), [f"Regla {i + 1}" for i in range(len(lifts))])
plt.xlabel("Lift")
plt.title("Lift de las reglas de asociación")
plt.show()
