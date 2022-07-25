import numpy
import numpy as np


def calcula_dist(p1, p2):
    """
    :param p1: primeiro ponto
    :param p2: segundo ponto
    :return: distância entre o primeiro e segundo ponto
    """
    if type(p1) == numpy.int64:
        return abs(p1 - p2)
    elif p1.shape[0] == 2:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)
    else:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2) ** (1 / 2)


def k_means_2(n_clusters, data):
    """
    :param n_clusters: número de clusters
    :param data: dados
    :return: dicionário onde a chave é o centróide e o valor associado os pontos próximos a ele
    """

    # Determinando a dimensão do array
    if len(data.shape) == 1:
        dimensao = 1
    elif data.shape[1] <= 3:
        dimensao = data.shape[1]
    else:
        return None

    # Obtendo centroides aleatórios
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        if dimensao == 1:
            cluster_centers = set([data[np.random.randint(0, data.shape[0])] for _ in range(n_clusters)])
        else:
            cluster_centers = set([tuple(data[np.random.randint(0, data.shape[0])]) for _ in range(n_clusters)])
    cluster_centers = tuple(cluster_centers)

    # Listas para saber se as médias dos pontos mudaram a cada iteração
    centers_before = None
    centers_after = []
    centers = None

    while centers_before != centers_after:
        centers_before = centers_after

        # Dicionário de pontos por centroides
        centers = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Associa cada ponto ao seu centroide mais próximo
        for ponto in data:
            min_center = None
            min_dist = np.inf
            for center in cluster_centers:
                dist = calcula_dist(ponto, center)
                if dist < min_dist:
                    min_center = center
                    min_dist = dist
            if dimensao == 1:
                centers[min_center].add(ponto)
            else:
                centers[min_center].add(tuple(ponto))

        # Cálculo os novos centroides
        centers_after = []
        for center in centers:
            ponto_medio = []
            for i in range(dimensao):
                if dimensao == 1:
                    soma = sum(centers[center])
                else:
                    soma = sum([center[i] for center in centers[center]])
                media = round(soma / len(centers[center]), 5)
                ponto_medio.append(media)
            centers_after.append(tuple(ponto_medio))

        cluster_centers = tuple(centers_after)
    return centers
        

# np.array([1, 2, 3, 4, 10, 15, 20, 100, 155, 200])
# np.array([[1, 2], [7, 11], [100, 1], [200, 4], [0, 0], [27, 33]])
# np.array([[1, 2, 100], [7, 11, 94], [100, 1, 16], [200, 4, 10], [0, 0, 3], [27, 33, 0]])

resultado4 = k_means_2(2, np.array([[1, 2, 100], [7, 11, 94], [100, 1, 16], [200, 4, 10], [0, 0, 3], [27, 33, 0]]))
print(resultado4)
