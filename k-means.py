import numpy as np


def k_means_1d(n_clusters, data):
    """
    :param n_clusters: número de clusters que o usuário deseja obter
    :param data: array de pontos de 1 dimensão
    :return: lista de centroides, lista dos conjuntos de pontos de acordo com os centroides
    """
    centers = None
    data.sort()

    # Evitando que haja centroides repetidos
    cluster_centers = set()
    while len(cluster_centers) != n_clusters:
        cluster_centers = set([np.random.randint(data[0], data[data.size - 1]) for _ in range(n_clusters)])
    cluster_centers = np.array(list(cluster_centers))

    # Listas para saber se as médias dos pontos mudaram
    list_before = None
    list_after = []

    while list_before != list_after:
        list_before = list_after
        centers = [set() for _ in range(n_clusters)]

        # Separando os pontos por centroide mais próximo
        for point in data:
            min_dist = np.inf
            min_center = None
            for i_center, center in enumerate(cluster_centers):
                dist = abs(point - center)
                if dist < min_dist:
                    min_dist = dist
                    min_center = i_center

            centers[min_center].add(point)

        # Atualizando os novos centroides
        list_after = []
        for center in centers:
            qtd_points = len(center)
            if qtd_points:
                list_after.append(round(sum(center) / qtd_points, 5))
            else:
                list_after.append(None)

    # TODO: mudar return para dicionário de arrays
    return list_after, centers


def geraPonto(x_lim, y_lim):
    eixo_x = np.random.randint(x_lim[0], x_lim[1])
    eixo_y = np.random.randint(y_lim[0], y_lim[1])
    return [eixo_x, eixo_y]


def calcula_dist(p1, p2):
    """
    :param p1: primeiro ponto (float x, float y)
    :param p2: segundo ponto (float x, float y)
    :return: distância entre o primeiro e segundo ponto
    """
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1 / 2)


def geraPonto3D(x_lim, y_lim, z_lim):
    eixo_x = np.random.randint(x_lim[0], x_lim[1])
    eixo_y = np.random.randint(y_lim[0], y_lim[1])
    eixo_z = np.random.randint(z_lim[0], z_lim[1])
    return [eixo_x, eixo_y, eixo_z]


def calculaDist3D(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** (1 / 2)


def k_means2D(n_clusters, data):
    pontos_X = data[:, 0]
    pontos_Y = data[:, 1]

    # Array de centroides aleatorios
    centroides = np.array([geraPonto((min(pontos_X), max(pontos_X)),
                                     (min(pontos_Y), max(pontos_Y)))
                           for _ in range(n_clusters)])
    print("centroides", centroides)
    # Lista dos centroides mais próximos por ponto
    lista_posterior = []
    lista_anterior = None

    while lista_anterior != lista_posterior:
        lista_anterior = lista_posterior

        lista_posterior = []
        # Verificando a qual dos novos centroides os pontos pertecncem agora
        for ponto in data:
            cent_min = 0
            dist_min = np.inf
            for i, centroide in enumerate(centroides):
                new_dist = calcula_dist(ponto, centroide)
                if new_dist < dist_min:
                    cent_min = i
                    dist_min = new_dist
            lista_posterior.append(cent_min)

        # Inicializando arrays locais que vão guardar as somas dos eixos dos
        # centroides e a quantidade de vezes que aparecem
        cent_x = np.zeros(len(centroides))
        cent_y = np.zeros(len(centroides))
        qtde_cent = np.zeros(len(centroides))

        # Somando nos eixos dos centroides e add um para cada vez que aparece na
        # lista
        for i, n in enumerate(lista_posterior):
            cent_x[n] += pontos_X[i]
            cent_y[n] += pontos_Y[i]
            qtde_cent[n] += 1

        # Atualizando os novos centroides
        for k in range(len(centroides)):
            centroides[k] = [round(cent_x[k] / qtde_cent[k], 5), round(cent_y[k] / qtde_cent[k], 5)]

    return lista_posterior, centroides


def k_means3D(n_clusters, data):
    qtd_pontos = data.shape[0]
    pontos_X = data[:, 0]
    pontos_Y = data[:, 1]
    pontos_Z = data[:, 1]


    ponto_X_ord = np.sort(pontos_X)
    ponto_Y_ord = np.sort(pontos_Y)
    ponto_Z_ord = np.sort(pontos_Z)

    x_min = ponto_X_ord[0]
    y_min = ponto_Y_ord[0]
    z_min = ponto_Y_ord[0]
    x_max = ponto_X_ord[qtd_pontos - 1]
    y_max = ponto_Y_ord[qtd_pontos - 1]
    z_max = ponto_Z_ord[qtd_pontos - 1]

    centroides = np.array([geraPonto3D((x_min, x_max), (y_min, y_max), (z_min, z_max)) for _ in range(n_clusters)])
    lista_lnk = []

    for ponto in data:
      cent = 0
      dist = calculaDist3D(ponto, centroides[0])

      for k in range(1,len(centroides)):
        new_dist = calculaDist3D(ponto, centroides[k])
        if new_dist < dist:
          cent = k
          dist = new_dist

      lista_lnk.append(cent)

    lista_anterior = None
    
    while lista_anterior != lista_lnk:
      lista_anterior = lista_lnk

      #Inicializando arrays locais qeu vão guardar as soma dos eixos dos
      #centroides e a qauntidade de vezes que aparecem
      cent_x = np.zeros(len(centroides))
      cent_y = np.zeros(len(centroides))
      cent_z = np.zeros(len(centroides))
      qtde_cent = np.zeros(len(centroides))

      #Somando nos eixos dos centroides e add um para cada vez que aparece na
      #lista
      for i,n in enumerate(lista_lnk):
        cent_x[n] += pontos_X[i]
        cent_y[n] += pontos_Y[i]
        cent_z[n] += pontos_Z[i]
        qtde_cent[n] += 1

      #Atualizando os novo centroides
      for k in range(len(centroides)):
        centroides[k] = [round(cent_x[k] / qtde_cent[k], 5), round(cent_y[k] / qtde_cent[k], 5), round(cent_z[k] / qtde_cent[k], 5)]

      lista_lnk = []
      #Verificando a qual dos novos centroides os pontos pertecncem agora
      for i,ponto in enumerate(data):
        cent = 0
        dist = calculaDist3D(ponto, centroides[0])

        for k in range(1,len(centroides)):
          new_dist = calculaDist3D(ponto, centroides[k])
          if new_dist < dist:
            cent = k
            dist = new_dist

        lista_lnk.append(cent)

    return lista_lnk, centroides


def k_means(n_clusters, data):
    if len(data.shape) == 1:
        return k_means_1d(n_clusters, data)
    elif data.shape[1] == 2:
        return k_means2D(n_clusters, data)
    elif data.shape[1] == 3:
        return k_means3D(n_clusters, data)
    else:
        return None
        

# resultado = k_means(3, np.array([1, 2, 3, 4, 10, 15, 20, 100, 155, 200]))
resultado = k_means(3, np.array([[1, 2], [7, 11], [100, 1], [200, 4], [0, 0], [27, 33]]))

print(resultado)
