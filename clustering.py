import numpy as np
from sklearn.neighbors import KDTree
import math
import random

def mean_shift(data,ancho_banda,iter,tol):
  tree=KDTree(data)
  puntos=np.copy(data)

  for i in range(iter):
    n_puntos=[]
    for punto in puntos:
      ind=tree.query_radius([punto],ancho_banda)[0]
      veci=data[ind]
      centroid=np.mean(veci,axis=0)
      n_puntos.append(centroid)
    n_puntos=np.array(n_puntos)

    if np.linalg.norm(n_puntos-puntos)<tol:
      break
    else:
      puntos=n_puntos
  return puntos


def kmeans(data, k, epochs, distance_threshold=0.1):
        
    def distance(v1, v2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(v1, v2)))    
    
    centroids = np.array(random.sample(list(data), k))
    for _ in range(epochs):       
        clusters = [np.argmin([distance(pixel, centroid) for centroid in centroids]) for pixel in data]        
        new_centroids = np.array([np.mean([data[i] for i in range(len(data)) if clusters[i] == j], axis=0)for j in range(k)])
        
        distances = [distance(a, b) for a, b in zip(centroids, new_centroids)]
        average_distance_change = sum(distances) / len(distances)        
        
        if average_distance_change < distance_threshold:
            break
        
        centroids = new_centroids 
    return centroids, clusters
