from featurex import extract_features_to_files, load_features_from_files
import os

if (
    not os.path.exists("x.txt")
    or not os.path.exists("y.txt")
    or not os.path.exists("z.txt")
):
    extract_features_to_files(limit=1000)

x, y, z = load_features_from_files()

print(x.shape)
print(y)
print(z)
