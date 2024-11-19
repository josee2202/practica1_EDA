#!/usr/bin/env python3

import numpy as np
import pandas as pd
from rich.console import Console    ## Necesarias para ver los tipos potenciales de variabels
from rich.table import Table

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


def analizar_variables(dataset):
    """
    Analiza las variables de un DataFrame y genera una tabla en consola con:
    - Nombre de la variable.
    - Tipo actual.
    - Tipo potencial (categoría, numérica, booleana, etc.).
    - Ejemplos de valores únicos.

    Utiliza la **** LIBRERÍA `rich` **** para una representación visual mejorada.
    """
    console = Console()
    table = Table(title="Análisis de Variables", show_header=True, header_style="bold cyan")
    table.add_column("Variable", style="bold white", no_wrap=True)
    table.add_column("Tipo Actual", style="dim white", no_wrap=True)
    table.add_column("Tipo Potencial", style="bold yellow", no_wrap=True)
    table.add_column("Ejemplos Únicos", style="bold green", no_wrap=False)

    for col in dataset.columns:
        dtype = str(dataset[col].dtypes)
        unique_values = dataset[col].dropna().unique()
        num_unique = len(unique_values)

        # Determinar el tipo actual
        if dtype.startswith('int'):
            tipo_actual = "int64"
        elif dtype.startswith('float'):
            tipo_actual = "float64"
        elif dtype == 'object':
            tipo_actual = "object"
        elif dtype == 'bool':
            tipo_actual = "bool"
        else:
            tipo_actual = "otro"

        # Determinar el tipo potencial
        if num_unique == len(dataset):  # ID si todos los valores son únicos
            tipo_potencial = "[bright_blue]ID[/bright_blue]"
        elif tipo_actual == "bool" or (num_unique == 2):
            tipo_potencial = "[bright_green]Booleana[/bright_green]"
        elif num_unique < 100:
            tipo_potencial = "[bright_yellow]Categoría[/bright_yellow]"
        else:
            tipo_potencial = "[bright_red]Numérica[/bright_red]"

        # Seleccionar algunos ejemplos de valores únicos
        ejemplos = ", ".join(map(str, unique_values[:5])) + ("..." if num_unique > 5 else "")

        table.add_row(col, tipo_actual, tipo_potencial, ejemplos)

    console.print(table)


def optimizar_tipo_longitud(df, col_type):
    for col in df.select_dtypes(include=[col_type]).columns:
        longitudes = df[col].dropna().astype(str).str.len()
        min_len = longitudes.min()
        max_len = longitudes.max()

        print(f"Columna: {col} | Longitud mínima: {min_len} | Longitud máxima: {max_len}")
        
        if max_len <= 5:  # Números pequeños
            df[col] = df[col].astype('int16')
            print(f" - Convertida a int16.")
        elif max_len <= 10:  # Números más grandes
            df[col] = df[col].astype('int32')
            print(f" - Convertida a int32.")
        else:
            print(f" - Se mantiene como int64.")



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



def calcular_porcentaje_target(df, columna):
    """
    Calcula el porcentaje de valores 1 en la columna TARGET
    para las filas donde una columna específica tiene valores NaN.

    Parámetros:
    - df (DataFrame): El dataframe que contiene los datos.
    - columna (str): El nombre de la columna a analizar.

    Retorna:
    - float: El porcentaje de valores 1 en la columna TARGET.
    """
    # Identificar las filas con valores NaN
    filas_nulas = df[columna].isnull()
    filas_nulas_df = df.loc[filas_nulas]
    
    # Contar los valores de TARGET
    target_counts = filas_nulas_df['TARGET'].value_counts()

    # Calcular el porcentaje de valores 1
    porcentaje_target = round((target_counts.get(1, 0) / target_counts.sum()) * 100, 2)
    
    return columna, porcentaje_target









