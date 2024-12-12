#!/usr/bin/env python3

import numpy as np
import pandas as pd
from rich.console import Console    ## Necesarias para ver los tipos potenciales de variabels
from rich.table import Table

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, balanced_accuracy_score,average_precision_score, precision_recall_curve, recall_score, precision_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report, make_scorer
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score, auc

#### NOTEBOOK 1 ####

## no entiendo poque no puedo usar funciones.py
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



### NOTEBOOK 2 #### 

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



def plot_feature(df, col_name, isContinuous, target):
    """
    Visualize a variable with and without faceting on the loan status.
    """
    # Configurar paleta de colores agradable y consistente
    sns.set_palette("Set2")  # Paleta general

    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), dpi=90)

    # Contar nulos antes de cualquier transformación
    count_null = df[col_name].isnull().sum()

    if isContinuous:
        # Usar colores consistentes para el histograma
        sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1, color="#5975A4")
    else:
        # Transformar a string para contar bien los NaN
        col_temp = df[col_name].copy()
        col_temp = col_temp.astype('str')  # Convertir todo a string para incluir NaN
        col_temp[df[col_name].isnull()] = 'NaN'  # Reemplazar valores nulos por 'NaN'

        # Usar un color fijo para el countplot
        sns.countplot(col_temp, order=sorted(col_temp.unique()), color="#8DC9A3", saturation=1, ax=ax1)

    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(f"{col_name} - Número de nulos: {count_null}")
    plt.xticks(rotation=90)

    if isContinuous:
        data_no_na = df[[col_name, target]].dropna()  # Excluye filas con NaN en las columnas relevantes
        # Usar una paleta agradable para el boxplot
        sns.boxplot(x=target, y=col_name, data=data_no_na, ax=ax2, palette="coolwarm")

        ax2.set_ylabel('')
        ax2.set_title(f"{col_name} by {target}")
    else:
        data = df.groupby(col_temp)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
        data.columns = [col_name, target, 'proportion']
        # Usar colores consistentes para el barplot
        sns.barplot(x=col_name, y='proportion', hue=target, data=data, saturation=1, ax=ax2, palette="husl")
        ax2.set_ylabel(f"{target} fraction")
        ax2.set_title(target)
        plt.xticks(rotation=90)
    ax2.set_xlabel(col_name)

    plt.tight_layout()



### Valores faltantes ###

def get_percent_null_values_target(pd_loan, list_var_continuous, target):

    pd_final = pd.DataFrame()
    for i in list_var_continuous:
        if pd_loan[i].isnull().sum()>0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]\
                                            .value_counts(normalize=True)).T
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum()/pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final.sort_values(by='sum_null_values', ascending=False)


def analyze_and_sort_columns(df, cont_cols, threshold=25):
    """
    Analiza la distribución de varias columnas continuas, calcula la media, mediana y moda,
    y ordena las gráficas en función de la diferencia entre media y mediana.
    También clasifica las columnas según si están por debajo o encima de un umbral.

    Muestra gráficos de histogramas para cada columna en el orden especificado.
    Las columnas con diferencias pequeñas entre media y mediana se grafican primero.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
        cont_cols (list): Lista de nombres de columnas continuas a analizar.
        threshold (float): Umbral para clasificar variables según la diferencia media-mediana.

    Retorna:
        dict: Diccionario con las métricas (media, mediana, moda) para cada columna, 
              ordenadas por la diferencia entre media y mediana.
        list: Lista de columnas recomendadas para imputar con la media.
        list: Lista de columnas recomendadas para imputar con la mediana.
    """
    results = {}
    differences = {}
    lista_aplicar_media = []
    lista_aplicar_mediana = []

    # Calcular métricas para cada columna
    for cont_col in cont_cols:
        if cont_col not in df.columns:
            print(f"La columna '{cont_col}' no se encuentra en el DataFrame. Se omitirá.")
            continue
        
        media = df[cont_col].mean()
        mediana = df[cont_col].median()
        moda = df[cont_col].mode()[0]
        
        # Guardar métricas
        results[cont_col] = {
            'mean': media,
            'median': mediana,
            'mode': moda
        }
        
        # Calcular diferencia entre media y mediana
        diferencia = abs(media - mediana)
        differences[cont_col] = diferencia

        # Clasificar según el umbral
        if diferencia / mediana * 100 <= threshold:
            lista_aplicar_media.append(cont_col)
        else:
            lista_aplicar_mediana.append(cont_col)

    # Ordenar columnas por la diferencia entre media y mediana
    sorted_columns = sorted(differences, key=differences.get)

    # Graficar en el orden establecido
    for cont_col in sorted_columns:
        metrics = results[cont_col]
        media, mediana, moda = metrics['mean'], metrics['median'], metrics['mode']
        
        plt.figure(figsize=(5, 2))  # Ajustar tamaño de las gráficas
        plt.hist(df[cont_col].dropna(), bins=10, alpha=0.6, edgecolor='black', label=f'{cont_col} Distribution')
        
        # Añadir líneas para la media, mediana y moda
        plt.axvline(media, color='blue', linestyle='--', linewidth=1.5, label=f'Mean: {media:.2f}')
        plt.axvline(mediana, color='orange', linestyle='--', linewidth=1.5, label=f'Median: {mediana:.2f}')
        plt.axvline(moda, color='green', linestyle='--', linewidth=1.5, label=f'Mode: {moda:.2f}')
        
        # Configuración del gráfico
        plt.title(f'{cont_col} Distribution (Sorted by Mean-Median Difference)')
        plt.xlabel(cont_col)
        plt.ylabel('Frequency')
        plt.legend(fontsize=7, loc='best', frameon=True)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    # Retornar las métricas calculadas, ordenadas
    return {col: results[col] for col in sorted_columns}, lista_aplicar_media, lista_aplicar_mediana



def crear_lista_tratamiento(df, cont_cols, threshold=25):
    """
    Clasifica columnas según la diferencia entre media y mediana y genera listas para el tratamiento.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
        cont_cols (list): Lista de nombres de columnas continuas a analizar.
        threshold (float): Umbral para clasificar variables según la diferencia media-mediana.

    Retorna:
        dict: Diccionario con las métricas (media, mediana, moda) para cada columna.
        list: Lista de columnas recomendadas para imputar con la media.
        list: Lista de columnas recomendadas para imputar con la mediana.
    """
    results = {}
    lista_aplicar_media = []
    lista_aplicar_mediana = []

    for cont_col in cont_cols:
        if cont_col not in df.columns:
            print(f"La columna '{cont_col}' no se encuentra en el DataFrame. Se omitirá.")
            continue

        media = df[cont_col].mean()
        mediana = df[cont_col].median()
        moda = df[cont_col].mode()[0]

        # Guardar métricas
        results[cont_col] = {
            'mean': media,
            'median': mediana,
            'mode': moda
        }

        # Calcular diferencia relativa entre media y mediana
        diferencia = abs(media - mediana)
        if diferencia / mediana * 100 <= threshold:
            lista_aplicar_media.append(cont_col)
        else:
            lista_aplicar_mediana.append(cont_col)

    return results, lista_aplicar_media, lista_aplicar_mediana



def get_deviation_of_mean_perc(pd_loan, list_var_continuous, target, multiplier):
    ### La dejo aquí porque me da problemas al pasarla al .py
    """
    Devuelve un DataFrame que muestra el porcentaje de valores que exceden el intervalo de confianza,
    junto con la distribución del target para esos valores.
    
    :param pd_loan: DataFrame con los datos a analizar.
    :param list_var_continuous: Lista de variables continuas a analizar.
    :param target: Variable objetivo para analizar la distribución de categorías.
    :param multiplier: Factor multiplicativo para determinar el rango de confianza (media ± multiplier * std).
    :return: DataFrame con las proporciones del target para los valores fuera del rango de confianza.
    """
    # DataFrame para almacenar los resultados finales
    pd_final = pd.DataFrame()

    # Obtener las categorías únicas del target para usarlas en las columnas
    target_categories = pd_loan[target].unique()

    # Iterar sobre cada variable continua de la lista
    for i in list_var_continuous:
        # Calcular la media y desviación estándar de la variable
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()

        # Determinar los límites del rango de confianza
        std_amp = multiplier * series_std
        left = series_mean - std_amp  # Límite inferior
        right = series_mean + std_amp  # Límite superior
        size_s = pd_loan[i].size  # Tamaño total de la serie (número de valores [filas])

        # Calcular el porcentaje de valores fuera del rango de confianza
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size / size_s

        # Si hay valores fuera del rango de confianza, analizar el target
        if perc_excess > 0:
            # Filtrar el target para los valores fuera del rango de confianza
            outliers = pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]
            
            # Calcular las proporciones de cada categoría del target
            proportions = outliers.value_counts(normalize=True)

            # Crear un DataFrame temporal con estas proporciones
            temp_df = proportions.to_frame().T.reset_index(drop=True)
            
            # Asignar nombres a las columnas basados en las categorías del target
            temp_df.columns = [str(cat) for cat in proportions.index]
            
            # Añadir información adicional al DataFrame temporal
            temp_df['variable'] = i  # Nombre de la variable continua
            temp_df['sum_outlier_values'] = outliers.size  # Total de valores fuera del rango
            temp_df['porcentaje_sum_null_values'] = perc_excess  # Porcentaje de valores fuera del rango

            # Concatenar los resultados de esta variable al DataFrame final
            pd_final = pd.concat([pd_final, temp_df], axis=0).reset_index(drop=True)

    # Si no se encontró ningún valor fuera del rango, mostrar mensaje
    if pd_final.empty:
        print('No existen variables con valores fuera del rango de confianza')
        
    return pd_final.sort_values(by='sum_outlier_values', ascending=False)



def get_corr_matrix(dataset=None, metodo='pearson', size_figure=[10, 8], threshold=0):
    """
    Genera una matriz de correlación y permite filtrar por un umbral sin mostrar valores exactos.
    
    Parámetros:
        dataset (pd.DataFrame): DataFrame para calcular la matriz de correlación.
        metodo (str): Método de correlación ('pearson', 'spearman', 'kendall').
        size_figure (list): Tamaño de la figura [ancho, alto].
        threshold (float): Umbral mínimo para mostrar correlaciones (en valor absoluto).
        
    Retorna:
        int: 0 si se ejecuta correctamente, 1 si hay algún error.
    """
    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1

    sns.set(style="white")
    
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo)
    
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    
    # Apply the threshold: set values below the threshold to NaN
    mask = abs(corr) < threshold
    corr = corr.where(~mask)
    
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    
    # Draw the heatmap without annotation
    sns.heatmap(corr, center=0, square=True, linewidths=.5, cmap='viridis', mask=mask, cbar=True, annot=False)
    
    plt.title(f"Matriz de Correlación ({metodo}, Threshold > {threshold})")
    plt.show()
    
    return 0



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


def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))


def cramers_v_heatmap_with_high_corr(df, categorical_columns, title="Matriz de Correlación (Cramér's V)", figsize=(10, 8)):
    """
    Calcula la matriz de correlación de Cramér's V, genera un mapa de calor y 
    muestra pares de variables con correlación entre 0.6 y 0.99.

    Parámetros:
        df (pd.DataFrame): DataFrame con las variables categóricas.
        categorical_columns (list): Lista de nombres de columnas categóricas.
        title (str): Título del gráfico.
        figsize (tuple): Tamaño del mapa de calor.
    """
    n = len(categorical_columns)
    matrix = pd.DataFrame(np.zeros((n, n)), columns=categorical_columns, index=categorical_columns)
    high_corr_pairs = []  # Lista para almacenar pares de variables con alta correlación
    
    for i, col1 in enumerate(categorical_columns):
        for j, col2 in enumerate(categorical_columns):
            if i == j:
                matrix.loc[col1, col2] = 1.0
            elif i < j:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                chi2 = chi2_contingency(confusion_matrix)[0]
                n_total = confusion_matrix.values.sum()
                phi2 = chi2 / n_total
                r, k = confusion_matrix.shape
                phi2corr = max(0, phi2 - ((k-1)*(r-1)) / (n_total-1))
                rcorr = r - ((r-1)**2) / (n_total-1)
                kcorr = k - ((k-1)**2) / (n_total-1)
                cramers_v_value = np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
                matrix.loc[col1, col2] = matrix.loc[col2, col1] = cramers_v_value
                
                # Almacenar pares con correlación alta
                if 0.6 <= cramers_v_value < 0.99:
                    high_corr_pairs.append((col1, col2, cramers_v_value))

    # Generar el mapa de calor
    plt.figure(figsize=figsize)
    sns.heatmap(matrix, annot=False, cmap="coolwarm", square=True, cbar=True)
    plt.title(title)
    plt.show()
    
    # Imprimir las variables con alta correlación
    if high_corr_pairs:
        print("\nPares de variables con alta correlación (entre 0.6 y 0.99):\n")
        for var1, var2, corr in high_corr_pairs:
            print(f"{var1} ↔ {var2}: {corr:.2f}")
    else:
        print("\nNo se encontraron pares de variables con correlación entre 0.6 y 0.99.")


def calculate_woe_iv(df, feature, target):
    """
    Calcula Weight of Evidence (WoE) e Information Value (IV) para una variable categórica.
    
    Parámetros:
        df (pd.DataFrame): Dataset que contiene la variable y el objetivo.
        feature (str): Nombre de la variable categórica.
        target (str): Nombre de la variable objetivo binaria (0: buenos, 1: malos).
    
    Retorna:
        pd.DataFrame: Tabla con WoE e IV por categoría.
        float: Valor total del IV para la variable.
    """
    # Contar buenos (0) y malos (1) por categoría
    feature_stats = df.groupby(feature)[target].agg(['count', 'sum'])
    feature_stats.columns = ['total', 'bad']
    feature_stats['good'] = feature_stats['total'] - feature_stats['bad']
    
    # Calcular porcentajes
    total_good = feature_stats['good'].sum()
    total_bad = feature_stats['bad'].sum()
    feature_stats['good_dist'] = feature_stats['good'] / total_good
    feature_stats['bad_dist'] = feature_stats['bad'] / total_bad
    
    # Calcular WoE
    feature_stats['WoE'] = np.log(feature_stats['good_dist'] / feature_stats['bad_dist'].replace(0, 1e-10))
    
    # Calcular IV
    feature_stats['IV'] = (feature_stats['good_dist'] - feature_stats['bad_dist']) * feature_stats['WoE']
    total_iv = feature_stats['IV'].sum()
    
    return feature_stats[['good_dist', 'bad_dist', 'WoE', 'IV']], total_iv



################# Evaluación modelos (03_.ipynb) ##################

def fast_eval(model, x_val, y_val, y_pred):
    """
    Evalúa rápidamente un modelo entrenado.
    
    Parámetros:
        model : modelo entrenado (RandomizedSearchCV o similar)
        x_val : array-like, Datos de entrada de validación o test
        y_val : array-like, Etiquetas reales de validación o test
        y_pred : array-like, Etiquetas predichas por el modelo
    """
    # Imprimir los mejores parámetros y el mejor score si es un RandomizedSearchCV o GridSearchCV
    if hasattr(model, "best_params_"):
        print("Mejores parámetros:", model.best_params_)
        print("Mejor score (validación cruzada):", model.best_score_)
    
    # Imprimir reporte de clasificación
    print("\nReporte de clasificación:")
    print(classification_report(y_val, y_pred))
    
    # Mostrar la matriz de confusión normalizada
    disp = ConfusionMatrixDisplay.from_estimator(
        model, x_val, y_val,
        cmap=plt.cm.Blues,
        normalize='true'
    )
    disp.ax_.set_title('Matriz de Confusión Normalizada')
    plt.show()

    # Imprimir la matriz de confusión numérica
    print("\nMatriz de confusión (normalizada):")
    print(disp.confusion_matrix)