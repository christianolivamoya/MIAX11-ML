# MIAX11-ML

Material preparado para las clases de Machine Learning del máster MIAX: Edición 11

Autores:

* Christian Oliva Moya

* Luis Fernando Lago Fernández

<hr>

## Contenido del repositorio

* `data` incluye algunos ficheros CSV para trabajar durante la parte práctica de las clases.

* `notebooks` incluye los notebooks que estamos usando como práctica durante las clases. En particular, un notebook tendrá el siguiente nombre: `<BLOQUE>_<N>_<DESCRIPCION>.ipynb`, donde `<BLOQUE>` será el bloque del temario que tiene relación con el notebook, `<N>` es simplemente un ordinal y `<DESCRIPCION>` da nombre al notebook. Por ejemplo, el notebook `3_1_decisiontree.ipynb` es el primer notebook del bloque 3.

* `slides` incluye las diapositivas que se están viendo durante las clases.

## Temario

* **Bloque 1: Introducción al curso y al Machine Learning. Conceptos básicos de ML**

* **Bloque 2: K-Nearest Neighbors (KNN). Introducción a Sklearn y Google Colab**
  * Notebook *2_1_knn*. Implementación manual de KNN. Comparación con `KNeighborsClassifier` de Sklearn.

* **Bloque 3: Árboles de decisión**
  * Notebook *3_1_decisiontree*. Implementación de `DecisionTreeClassifier` y visualización del árbol con Sklearn.
  * Notebook *3_2_decisiontree_bankrupt*. Implementación con Sklearn.

* **Bloque 4: Clustering**
  * Notebook *4_1_clustering_aglomerativo*. Implementación manual de clustering aglomerativo con centroid-link. Comparación con Scipy.
  * Notebook *4_2_clustering_kmeans*. Clustering basado en centroides. Implementación de `KMeans` con Sklearn.
  * Notebook *4_3_clustering_em*. Clustering basado en mezcla de Gaussianas. Implementación de `GassuianMixture` con Sklearn.
  * Notebook *4_4_comparacion_clustering*. Dos problemas sintéticos para comparar distintos métodos de clustering.
  * Notebook *4_5_clustering_financiero_con_momentum*. Planteamiento desde otro prisma para hacer clustering de activos.

* **Bloque 5: Preprocesado**
  * Notebook *5_1_exploracion*. Visualización y limpieza de valores erróneos.
  * Notebook *5_2_transformaciones*. Transformaciones de los datos: ordinales, binarios, categóricos, etc.
  * Notebook *5_3_estandarizacion*. Normalización de los datos: `StandardScaler` y `MinMaxScaler` con Sklearn.

* **Bloque 6: Métricas de evaluación** 
  * Notebook *6_1_cross_validation_and_threshold*. Validación cruzada para selección de K y el umbral de decisión con KNN.

* **Bloque 7: Reducción de dimensionalidad**
  * Notebook *7_1_feature_importance*. Implementación de la importancia por permutación usando `permutation_importance` de Sklearn.
  * Notebook *7_2_pca*. Implementación de `PCA` usando Sklearn. Análisis de la varianza explicada.

* **Bloque 8: Clasificación Bayesiana**
  * Notebook *8_1_estimacion_densidades*. Uso de diversos métodos para estimar densidades. Clasificación Bayesiana a partir de las densidades estimadas. Naive Bayes.

* **Bloque 9: Modelos lineales**
  * Notebook *9_1_modelos_lineales*. Regresión lineal. Complejidad. Dilema bias-varianza. Regularización.

* **Bloque 10: Métodos de Kernel**
  * Notebook *10_1_kernel_methods*. Introducción a los métodos de kernel.
  * Notebook *10_2_kernel_ridge*. Regresión lineal basada en kernels.

* **Bloque 11: Support Vector Machines**
  * Notebook *11_1_support_vector_machines*. Ejemplos sencillos con problemas en 2D.
  * Notebook *11_2_svm_pima*. Aplicación de SVMs al problema Pima Indians Diabetes.



