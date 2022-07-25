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


def geraPonto3D(x_lim, y_lim, z_lim):
    eixo_x = np.random.randint(x_lim[0], x_lim[1])
    eixo_y = np.random.randint(y_lim[0], y_lim[1])
    eixo_z = np.random.randint(z_lim[0], z_lim[1])
    return [eixo_x, eixo_y, eixo_z]


def k_means_1d(n_clusters, data):
    """
    :param n_clusters: número de clusters que o usuário deseja obter
    :param data: array de pontos de 1 dimensão
    :return: lista de centroides, lista dos conjuntos de pontos de acordo com os centroides
    """
    # Evitando que haja centroides repetidos
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        cluster_centers = set([data[np.random.randint(0, data.size)] for _ in range(n_clusters)])
    cluster_centers = np.array(list(cluster_centers))

    # Listas para saber se as médias dos pontos mudaram
    centers_before = None
    centers_after = []
    centers = None

    while centers_before != centers_after:
        centers_before = centers_after
        centers = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Separando os pontos por centroide mais próximo
        for point in data:
            min_dist = np.inf
            min_center = None
            for center in cluster_centers:
                dist = abs(point - center)
                if dist < min_dist:
                    min_dist = dist
                    min_center = center
            centers[min_center].add(point)

        # Atualizando os novos centroides
        centers_after = []
        for center in centers:
            qtd_points = len(centers[center])
            if qtd_points:
                centers_after.append(round(sum(centers[center]) / qtd_points, 5))
            else:
                centers_after.append(None)
        cluster_centers = centers_after

    return centers


def k_means_2d(n_clusters, data):
    # Array de centroides aleatorios
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        cluster_centers = set([tuple(data[np.random.randint(0, data.shape[0])]) for _ in range(n_clusters)])
    cluster_centers = tuple(cluster_centers)

    # Listas para saber se as médias dos pontos mudaram
    centers_before = None
    centers_after = []
    centers = None

    while centers_before != centers_after:
        centers_before = centers_after
        centers = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Verificando a qual dos novos centroides os pontos pertencem agora
        for ponto in data:
            min_center = None
            min_dist = np.inf
            for center in cluster_centers:
                dist = calcula_dist(ponto, center)
                if dist < min_dist:
                    min_center = center
                    min_dist = dist
            centers[min_center].add(tuple(ponto))

        centers_after = []
        for center in centers:
            soma_x = sum([center[0] for center in centers[center]])
            soma_y = sum([center[1] for center in centers[center]])
            media_x = round(soma_x / len(centers[center]), 5)
            media_y = round(soma_y / len(centers[center]), 5)
            centers_after.append((media_x, media_y))

        cluster_centers = centers_after

    return centers


def k_means_3d(n_clusters, data):
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        cluster_centers = set([tuple(data[np.random.randint(0, data.shape[0])]) for _ in range(n_clusters)])
    cluster_centers = tuple(cluster_centers)

    # Listas para saber se as médias dos pontos mudaram
    centers_before = None
    centers_after = []
    centers = None

    while centers_before != centers_after:
        centers_before = centers_after
        centers = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Verificando a qual dos novos centroides os pontos pertencem agora
        for ponto in data:
            min_center = None
            min_dist = np.inf
            for center in cluster_centers:
                dist = calcula_dist(ponto, center)
                if dist < min_dist:
                    min_center = center
                    min_dist = dist
            centers[min_center].add(tuple(ponto))

        centers_after = []
        for center in centers:
            soma_x = sum([center[0] for center in centers[center]])
            soma_y = sum([center[1] for center in centers[center]])
            soma_z = sum([center[2] for center in centers[center]])
            media_x = round(soma_x / len(centers[center]), 5)
            media_y = round(soma_y / len(centers[center]), 5)
            media_z = round(soma_z / len(centers[center]), 5)
            centers_after.append((media_x, media_y, media_z))

        cluster_centers = centers_after

    return centers


def k_means_2(n_clusters, data):
    if len(data.shape) == 1:
        dimensao = 1
    elif data.shape[1] <= 3:
        dimensao = data.shape[1]
    else:
        return None

    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        if dimensao == 1:
            cluster_centers = set([data[np.random.randint(0, data.shape[0])] for _ in range(n_clusters)])
        else:
            cluster_centers = set([tuple(data[np.random.randint(0, data.shape[0])]) for _ in range(n_clusters)])
    cluster_centers = tuple(cluster_centers)

    # Listas para saber se as médias dos pontos mudaram
    centers_before = None
    centers_after = []
    centers = None

    while centers_before != centers_after:
        print(cluster_centers)
        centers_before = centers_after
        centers = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Verificando a qual dos novos centroides os pontos pertencem agora
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
        print("centers", centers)

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


def k_means(n_clusters, data):
    if len(data.shape) == 1:
        return k_means_1d(n_clusters, data)
    elif data.shape[1] == 2:
        return k_means_2d(n_clusters, data)
    elif data.shape[1] == 3:
        return k_means_3d(n_clusters, data)
    else:
        return None
        

# np.array([1, 2, 3, 4, 10, 15, 20, 100, 155, 200])
# np.array([[1, 2], [7, 11], [100, 1], [200, 4], [0, 0], [27, 33]])
# np.array([[1, 2, 100], [7, 11, 94], [100, 1, 16], [200, 4, 10], [0, 0, 3], [27, 33, 0]])

resultado4 = k_means_2(2, np.array([[1, 2, 100], [7, 11, 94], [100, 1, 16], [200, 4, 10], [0, 0, 3], [27, 33, 0]]))
print(resultado4)
