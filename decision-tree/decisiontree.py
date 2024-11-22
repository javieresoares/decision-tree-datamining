# Import library yang diperlukan
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, tree
import matplotlib.pyplot as plt
import pydotplus
from IPython.display import Image
import numpy as np
import pandas as pd

# Load dataset iris
iris = datasets.load_iris()
features = iris['data']
target = iris['target']

# Definisikan model Decision Tree
decisiontree = DecisionTreeClassifier(
    random_state=0,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0,
)

# Fit model pada data iris
model = decisiontree.fit(features, target)

# Prediksi pada observasi baru
observation = [[5, 4, 3, 2]]
print("Prediksi kelas:", model.predict(observation))
print("Probabilitas kelas:", model.predict_proba(observation))

# Visualisasi Decision Tree
dot_data = tree.export_graphviz(
    decisiontree,
    out_file=None,
    feature_names=iris['feature_names'],
    class_names=iris['target_names'],
    filled=True,  # Tambahkan warna untuk mempermudah visualisasi
)

graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())
graph.write_png("iris.png")

# ============================================
# Membaca dataset dari file CSV
irisDataset = pd.read_csv('dataset-iris.csv', delimiter=',', header=0)

# Mengubah kelas (kolom "Species") dari string ke unique-integer
irisDataset["Species"] = pd.factorize(irisDataset.Species)[0]

# Menghapus kolom "Id" jika ada
if "Id" in irisDataset.columns:
    irisDataset = irisDataset.drop(labels="Id", axis=1)

# Mengubah dataframe ke array numpy
irisDataset = irisDataset.to_numpy()

# Membagi dataset: 40 baris untuk training, 20 untuk testing
dataTraining = np.concatenate(
    (irisDataset[0:40, :], irisDataset[50:90, :]), axis=0
)
dataTesting = np.concatenate(
    (irisDataset[40:50, :], irisDataset[90:100, :]), axis=0
)

# Memisahkan dataset menjadi input dan label
inputTraining = dataTraining[:, 0:4]
inputTesting = dataTesting[:, 0:4]
labelTraining = dataTraining[:, 4]
labelTesting = dataTesting[:, 4]

# Mendifinisikan Decision Tree Classifier
model = DecisionTreeClassifier()

# Melatih model
model = model.fit(inputTraining, labelTraining)

# Memprediksi data testing
hasilPrediksi = model.predict(inputTesting)

# Evaluasi hasil prediksi
print("Label sebenarnya:", labelTesting)
print("Hasil prediksi:", hasilPrediksi)

# Menghitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()

print("Prediksi benar:", prediksiBenar, "data")
print("Prediksi salah:", prediksiSalah, "data")
print("Akurasi:", (prediksiBenar / (prediksiBenar + prediksiSalah)) * 100, "%")
