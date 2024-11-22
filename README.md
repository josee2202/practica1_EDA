

Objetivos del Proyecto
El objetivo principal de este proyecto es determinar la probabilidad de insolvencia de un cliente a la hora de concederle un préstamo. Esto se logra a través de un análisis exhaustivo de los datos disponibles, el desarrollo de modelos predictivos y la evaluación de su rendimiento. Aunque el alcance final incluye la creación de modelos, el foco inicial de este proyecto está en el Análisis Exploratorio de Datos (EDA), que sienta las bases para las siguientes etapas.

Objetivos Específicos
Entender los Datos:

Familiarizarse con el conjunto de datos, sus variables y su distribución.
Comprender el significado y la relevancia de cada variable en relación con la variable objetivo: la insolvencia del cliente.
Análisis Exploratorio de Datos (EDA):

Identificar patrones, tendencias y relaciones dentro del dataset.
Evaluar las distribuciones de las variables categóricas y numéricas, utilizando herramientas visuales y estadísticas.
Investigar posibles correlaciones entre variables, especialmente en relación con la variable objetivo, para identificar qué características podrían influir en la insolvencia.
Detección y Tratamiento de Problemas en los Datos:

Identificar y manejar valores nulos mediante estrategias como imputación.
Detectar y analizar valores atípicos (outliers) que puedan impactar negativamente el análisis y los modelos.
Evaluar la existencia de colinealidad entre variables que puedan redundar en problemas durante la etapa de modelado.
Preparación para el Modelado:

Clasificar las variables en función de su tipo (categóricas, booleanas, continuas) y tratarlas adecuadamente.
Codificar las variables categóricas utilizando técnicas como One-Hot Encoding, Target Encoding y Label Encoding.
Estandarizar las variables numéricas para asegurar que todas tengan un impacto equitativo en los modelos.
Definición de una Hipótesis Inicial:

Basándose en el análisis del EDA, desarrollar un perfil preliminar del cliente con mayor probabilidad de insolvencia, considerando factores como nivel educativo, tipo de vivienda, situación laboral, entre otros.
Reflexión sobre el Dataset y el Problema:

Concluir si los datos actuales permiten abordar el problema de manera efectiva.
Reconocer los desafíos que plantea el desbalanceo de la variable objetivo y planificar cómo abordarlo en futuras etapas.
En resumen, el EDA es fundamental para construir una base sólida que permita seleccionar características relevantes y guiar las decisiones en el modelado. Además, proporciona insights iniciales que facilitan la generación de hipótesis y el entendimiento del comportamiento de los clientes en relación con la insolvencia.



Análisis Exploratorio de Datos (EDA):
Comprensión del dataset: Análisis inicial para entender la naturaleza de las variables y su distribución.
Distribución y correlaciones: Uso de herramientas visuales para estudiar la dispersión de las variables y su relación con el objetivo.
Valores nulos y atípicos: Identificación y manejo de datos faltantes y valores extremos.
Transformaciones y codificaciones: Preparación de variables categóricas mediante técnicas como One-Hot Encoding y Target Encoding.
Normalización: Escalado de variables numéricas para garantizar consistencia en los modelos.
Modelado Predictivo:
Preparación del dataset: División en conjuntos de entrenamiento y prueba de manera estratificada.
Selección y comparación de modelos: Pruebas con diferentes algoritmos de clasificación para encontrar el más efectivo.
Manejo del desbalanceo: Uso de técnicas como sobremuestreo o submuestreo para equilibrar las clases en el dataset.
Optimización: Validación cruzada y ajuste de hiperparámetros para maximizar el rendimiento.
Explicabilidad del modelo: Evaluación del impacto de las variables más relevantes en las predicciones.


Estructura del Repositorio
El repositorio del proyecto está organizado en las siguientes carpetas:

data/: Contiene los datos usados en el proyecto.

	variables/: Aquí el almacenado listas para realizar trataminetos.

src/: Código con las funciones usadas.

notebooks/

html/: Reportes exportados en formato HTML.

env/: Archivo con los requerimientos del entorno para ejecutar el proyecto sin problemas.

