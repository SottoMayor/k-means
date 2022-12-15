import numpy as np  # para manipular os vetores
from matplotlib import pyplot as plt  # para plotar os gráficos
from sklearn.cluster import KMeans  # para usar o KMeans

# matriz com as coordenadas geográficas do McDonald's
dataset_MC = np.array(
    [[-3.09778, -60.01229],
     [-3.10432, -60.02323],
        [-3.09337, -60.02214],
        [-3.10105, -60.01133],
        [-3.07952, -60.07181]])

# matriz com as coordenadas geográficas do Burger King
dataset_BK = np.array([
    [-3.11856, -59.98207],
    [-3.09077, -60.00876],
    [-3.09194, -60.02219],
    [-3.10105, -60.01133],
    [-3.04665, -60.07999],
    [-3.10094, -60.02384],
    [-3.02744, -59.97708],
    [-3.08329, -60.07215],
    [-2.99563, -60.00292]])

kmeans = KMeans(n_clusters=3,  # numero de clusters
                init='k-means++', n_init=10,
                max_iter=300)  # numero máximo de iterações
pred_y = kmeans.fit_predict(dataset_BK)

# posicionamento dos eixos x e y
plt.scatter(dataset_BK[:, 1], dataset_BK[:, 0], c=pred_y)
plt.xlim(-60.15, -59.85)
plt.ylim(-3.25, -2.75)
plt.grid(True, 'both', 'both')
plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[
            :, 0], s=70, c='red')
plt.show()
