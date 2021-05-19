import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
dataset_path = "/media/krukts/HDD/BioDiploma/BalancedFEData/HISTO_Image_Dataset_CAMELYON16_3_Classes_18K_Tiles"
json_file = "Test2_BalancedNormalized.json"
if __name__ == '__main__':
    all_names = []
    all_labels = []
    all_predicted = []
    all_wsi_indexes = []
    all_features = []

    with open(os.path.join(dataset_path, json_file), 'r') as file_in:
        all_data = json.load(file_in)

    for key in all_data:
        print(key)
        all_names.append(key)
        all_labels.append(all_data[key]["label"])
        all_predicted.append(all_data[key]["predicted"])
        all_features.append(all_data[key]["features"])
        all_wsi_indexes.append(all_data[key]["wsi_index"])

    labels_np = np.array(all_labels)
    n_clusters = 2

    print("Our FE NN accuracy: 0.9156")
    # KMeans
    k_means = KMeans(n_clusters=n_clusters, random_state=1)
    k_means.fit(np.asarray(all_features))
    clustersKmeans, centers = k_means.labels_, k_means.cluster_centers_
    # print(clustersKmeans)
    print("KMeans accuracy in clusterization: ", (labels_np == np.array(clustersKmeans)).sum() / len(all_data))

    # SVM
    svc = svm.SVC(kernel="rbf")
    svc.fit(np.asarray(all_features)[::6, :], labels_np[::6])
    predicted_svc = svc.predict(np.asarray(all_features))
    print("SVC (kernel=rbf) accuracy in clusterization: ", (labels_np == np.array(predicted_svc)).sum() / len(all_data))

    #RandomForest
    random_forest=RandomForestClassifier(max_depth=5)
    random_forest.fit(np.asarray(all_features)[::6, :], labels_np[::6])
    predicted_rf=random_forest.predict(np.asarray(all_features))
    print("RandomForestClassifier accuracy in clusterization: ", (labels_np == np.array(predicted_rf)).sum() / len(all_data))

    print("Script worked!")
