import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets
import pydotplus
from IPython.display import Image

# Debug: Tambahkan Graphviz PATH ke lingkungan sistem secara manual
os.environ["PATH"] += os.pathsep + "C:/Program Files/Graphviz/bin"

# Debug: Verifikasi apakah Graphviz dapat dijalankan
if os.system("dot -version") != 0:
    print("Error: Graphviz tidak ditemukan di PATH!")
else:
    print("Graphviz ditemukan.")

# ===== Bagian 1: Visualisasi Model Decision Tree dengan Dataset Iris =====

# Load dataset iris
iris = datasets.load_iris()
features = iris['data']
target = iris['target']

# Buat model Decision Tree
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
dot_data = export_graphviz(
    decisiontree,
    out_file=None,
    feature_names=iris['feature_names'],
    class_names=iris['target_names'],
    filled=True,
    rounded=True,
    special_characters=True,
)

# Konversi dot data ke grafik
graph = pydotplus.graph_from_dot_data(dot_data)

# Tampilkan grafik sebagai gambar
Image(graph.create_png())
graph.write_png("iris.png")  # Simpan grafik ke file

print("Visualisasi Decision Tree berhasil disimpan sebagai 'iris.png'.")

# ===== Bagian 2: Dataset Custom =====

# Membaca dataset custom dari file CSV
# Pastikan file 'dataset-iris.csv' berada di direktori yang sama
irisDataset = pd.read_csv('dataset-iris.csv', delimiter=',', header=0)

# Mengubah kelas (kolom "Species") dari string ke unique integer
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

# Definisikan model Decision Tree Classifier
custom_model = DecisionTreeClassifier()

# Training model dengan data custom
custom_model = custom_model.fit(inputTraining, labelTraining)

# Prediksi data testing
hasilPrediksi = custom_model.predict(inputTesting)

# Evaluasi hasil prediksi
print("Label sebenarnya:", labelTesting)
print("Hasil prediksi:", hasilPrediksi)

# Menghitung akurasi
prediksiBenar = (hasilPrediksi == labelTesting).sum()
prediksiSalah = (hasilPrediksi != labelTesting).sum()

print("Prediksi benar:", prediksiBenar, "data")
print("Prediksi salah:", prediksiSalah, "data")
print("Akurasi:", (prediksiBenar / (prediksiBenar + prediksiSalah)) * 100, "%")
