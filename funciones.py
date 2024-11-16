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


import os
print(os.getcwd())




