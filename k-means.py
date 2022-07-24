import numpy as np


def k_means1D(n_clusters, data):
    points_by_center = None

    data.sort()

    # Evitando que haja centróides repetidos
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        cluster_centers = set([np.random.randint(data[0], data[data.size - 1]) for _ in range(n_clusters)])
    cluster_centers = np.array(list(cluster_centers))

    lista_anterior = None
    lista_atual = []

    while lista_anterior != lista_atual:
        lista_anterior = lista_atual
        points_by_center = dict(zip(cluster_centers, [set() for _ in range(n_clusters)]))

        # Separando os pontos por centróide mais próximo
        for point in data:
            min_dist = np.inf
            min_center = None
            for center in points_by_center:
                dist = abs(point - center)
                if dist < min_dist:
                    min_dist = dist
                    min_center = center

            points_by_center[min_center].add(point)

        # Coletando a média dos pontos de cada centróide
        lista_atual = []
        for center in points_by_center:
            sum_points = sum(points_by_center[center])
            qtd_points = len(points_by_center[center])
            if qtd_points:
                lista_atual.append(round(sum_points / qtd_points, 5))
            else:
                lista_atual.append(None)

    return points_by_center


def k_means(n_clusters, data):
    if len(data.shape) == 1:
        return k_means1D(n_clusters, data)
    elif data.shape[1] == 2:
        pass
    elif data.shape[1] == 3:
        pass
    else:
        return None
        

resultado = k_means(3, np.array([1, 2, 3, 4, 10, 15, 20, 100, 155, 200]))
print(resultado)
