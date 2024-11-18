#!/usr/bin/env python3

import numpy as np
import pandas as pd

def dame_variables_categoricas(dataset=None, max_unique_values = 100):
    '''
    ----------------------------------------------------------------------------------------------------------
    Función dame_variables_categoricas:
    ----------------------------------------------------------------------------------------------------------
        -Descripción: Función que recibe un dataset y devuelve una lista con los nombres de las 
        variables categóricas
        -Inputs: 
            -- dataset: Pandas dataframe que contiene los datos
        -Return:
            -- lista_var_categoricas: lista con los nombres de las variables categóricas del
            dataset de entrada con menos de 100 valores diferentes
            -- 1: la ejecución es incorrecta
    '''
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    lista_var_categoricas = []
    lista_var_altas_dimensiones = []
    for i in dataset.columns:
        if (dataset[i].dtype not in [float, int]):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < max_unique_values:
                lista_var_categoricas.append(i)
            else:
                lista_var_altas_dimensiones.append(i)

    return lista_var_categoricas, lista_var_altas_dimensiones

def graficar_nan_distribucion_target(df, col_name):
    """
    Grafica la distribución de TARGET para las filas con NaN en col_name.
    """
    try:
        # Identificar las filas con valores NaN
        filas_nulas = df[col_name].isnull()
        filas_nulas_df = df.loc[filas_nulas]

        if filas_nulas_df.empty:
            print(f"La columna {col_name} no tiene valores NaN.")
            return

        plt.figure(figsize=(6, 4))
        sns.countplot(data=filas_nulas_df, x='TARGET', palette='pastel', edgecolor='black')

        # Añadir etiquetas y título
        plt.title(f'Distribución de TARGET en filas con {col_name} nulo', fontsize=16)
        plt.xlabel('TARGET', fontsize=14)
        plt.ylabel('Count', fontsize=14)

        # Calcular el total de filas con valores nulos
        total = len(filas_nulas_df)

        # Obtener el conteo de cada categoría en TARGET
        counts = filas_nulas_df['TARGET'].value_counts()

        # Añadir porcentajes como texto en el gráfico
        for i, (category, value) in enumerate(counts.items()):
            percentage = f"{(value / total) * 100:.2f}%"  # Calcular porcentaje
            plt.text(i, value , percentage, ha='center', fontsize=6, color='black')  # Ajustar posición de texto

        plt.show()

    except Exception as e:
        print(f"Error al procesar la columna {col_name}: {e}")



import os
print(os.getcwd())




