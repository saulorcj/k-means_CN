import enum
import numpy as np

def k_means1D(n_clusters, data):
    data = np.sort(data)

    min = data[0]
    max = data[data.size - 1]

    centroides = np.array([np.random.randint(min, max) for _ in range(n_clusters)])
    lista_anterior  = list(centroide)
    lista = []

    #Consertar erro
    dicionario = dict(((centroide, set()) for centroide in centroides))
    print(dicionario)

    for i, ponto in enumerate(data):
        min_dist = np.inf
        min_centr = None

        for key in dicionario:
            if abs(ponto - key) < min_dist:
                min_dist = ponto - key
                min_centr = key

        dicionario[min_centr].add(i)

    for k in dicionario:
        lista.append(sum([data[index] for index in dicionario[k]]) / len(dicionario[k]))
    

    while(lista_anterior != lista):

    for k in dicionario:
        lista.append(sum([data[index] for index in dicionario[k]]) / len(dicionario[k]))



def k_means(n_clusters, data):
    if len(data.shape) == 1:
        k_means1D(n_clusters, data)
    elif data.shape[1] == 2:
        pass
    elif data.shape[1] == 3:
        pass
    else:
        print("erro")
        

k_means(2, np.array([1,2,3,4,10]))
