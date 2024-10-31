import numpy as np
from sklearn.neighbors import NearestNeighbors

def pca(data,dim):

    d2=data-np.mean(data,axis=0)
    cov_ma=np.cov(d2,rowvar=False)

    eigenvalores,eigenvectores=np.linalg.eig(cov_ma)

    i_or=np.argsort(eigenvalores[::-1])
    eigen_val=eigenvalores[i_or]
    eigen_vec=eigenvectores[:,i_or]

    eigen_vec_2=eigen_vec[:,:dim]

    d3=np.dot(d2,eigen_vec_2)
    return d3

def umap(data, dim,n_neighbors,epoch,l_r):
  knn=NearestNeighbors(n_neighbors=n_neighbors)
  knn.fit(data)
  distancias,indices=knn.kneighbors(data)

  n=data.shape[0]
  g=np.zeros((n,n))

  for i in range(n):
    for j in range(1,n_neighbors):
      g[i,indices[i,j]]=np.exp((-distancias[i,j])**2)

  g=g/g.sum(axis=1,keepdims=True)
  Y=np.random.rand(n,dim)

  for i in range(epoch):
    d_y=np.linalg.norm(Y[:,np.newaxis]-Y,axis=2)
    #probabilidades y normalizacion
    p_Y=np.exp(-d_y**2)
    p_Y=p_Y/p_Y.sum(axis=1,keepdims=True)

    #gradiente
    grad=np.zeros_like(Y)
    for j in range(n):
      for k in range(n):
        grad[j]=grad[j]+g[j,k]-p_Y[j,k]
    Y=Y+grad*l_r
  return Y
