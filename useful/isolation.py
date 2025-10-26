# Ejemplo mostrando cómo predecir con IsolationForest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Generar los datos simulados
n_samples = 100 # Número de muestras por grupo
n_outliers = 20 # Número de anomalías
rng = np.random.RandomState(42) # Semilla para la reproducibilidad

# Generar dos grupos normales con distribución gaussiana. n_samples filas con 2 columnas
# a las que se suma el array [2,2] o [-2,-2]
group_1 = rng.randn(n_samples, 2) + np.array([2, 2])
group_2 = rng.randn(n_samples, 2) + np.array([-2, -2])

# Generar algunos puntos anómalos con distribución uniforme
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

# Concatenar los datos
X = np.concatenate([group_1, group_2, outliers])

# Entrenar el modelo de IsolationForest con los datos
clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X)

# Hacer predicciones con el modelo entrenado
y_pred = clf.predict(X)

# 1 valores normales, -1 anomalías
print(y_pred)

# Visualizar los resultados
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=20, edgecolor='k')
plt.title('Predicciones de IsolationForest')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
