import csv, argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("vector_file", help="Name of the (un-noised) .txt to use")
args = parser.parse_args()

data = []
labels = []
with open("Vectors/" + args.vector_file + ".csv", 'r') as vector_file:
    reader = csv.reader(vector_file)
    for row in reader:
        data.append(row[:-1])
        labels.append(row[-1])


pca = PCA(n_components = 50)
tsne = TSNE(n_components = 2)

pca_data = pca.fit_transform(data)
tsne_data = tsne.fit_transform(pca_data)


plt.scatter(tsne_data[:,0], tsne_data[:,1], c = labels)
plt.show()
