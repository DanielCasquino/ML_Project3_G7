from featurex import extract_features_to_files, load_features_from_files
import os
import numpy as np
from dimred import pca, umap
from clustering import kmeans, mean_shift


np.random.seed(42)

# first we check if the data is already there
if (
    not os.path.exists("x.csv")
    or not os.path.exists("y.csv")
    or not os.path.exists("z.csv")
):
    extract_features_to_files(limit=1000)

# load features, labels, and label names from the csvs
x, y, z = load_features_from_files()

# normalization cause pca suffers otherwise
normalized_matrix = (x - np.min(x, axis=0)) / ( np.max(x, axis=0) - np.min(x, axis=0))

# dimensionality reduction
x_pca = pca(x, 200)
# x_umap = umap(data=x, dim=200, n_neighbors=4, epoch=100, l_r=0.1)


print(np.array(x_pca).shape)

centroids, clusters = kmeans(x_pca, 10, 100)
pts = mean_shift(x_pca, 0.1, 100, 0.1)


# kmeans visualization (3 features)
x_pca_3d = pca(x_pca, 3)
centroids_3d = pca(centroids, 3)
