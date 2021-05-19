import math

import torch
import os
import cv2
from FeatureExtraction.model import FEModel
from FeatureExtraction.dataset import CAM17Dataset
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import histomicstk.preprocessing.color_normalization as hist

shuffle_dataset = True
save_path = "/media/krukts/HDD/BioDiploma/FeatureExtraction/log"
# TODO: Check this!!!!!!!!!!!!!
save_name = "BalancedNormalized"

dataset_path = "/media/krukts/HDD/BioDiploma/BalancedFEData/HISTO_Image_Dataset_CAMELYON16_3_Classes_18K_Tiles"
img_size = (224, 224)


def calc_all_dist_prob(examples, anchor):
    dist = np.zeros(examples.shape[0])
    for i in range(examples.shape[0]):
        # L2
        # dist[i] = torch.sqrt(torch.sum((anchor - examples[i]) ** 2)).item()
        # L1
        dist[i] = torch.sum(torch.abs(anchor - examples[i])).item()

    # dist /= np.max(dist)
    # dist = 1 - dist
    return dist


if __name__ == '__main__':
    model = FEModel()
    model.load_state_dict(torch.load(os.path.join(save_path, save_name)))
    print("Model was loaded: ", save_name)

    model.to("cpu")
    print(model)

    criterion = torch.nn.CrossEntropyLoss()

    # Dataset creation
    dataset = CAM17Dataset(type="test", dataset_path=dataset_path, img_size=img_size)
    train_dataloader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=True,
                                  drop_last=True)

    i = 0
    m = 81
    sq = int(math.sqrt(m))
    s = 224

    all_features = torch.zeros((m, 256))
    all_images = torch.ones((m, 3, 224, 224))
    all_labels = torch.ones(m)
    all_predicted = torch.ones(m)
    total_image = np.zeros((sq * s, sq * s, 3))

    with torch.no_grad():
        model.eval()
        for batch, labels in train_dataloader:
            print(i)
            outputs = model(batch)

            all_predicted[i] = int(torch.softmax(outputs, 1)[0, 1].item() > 0.5)
            all_images[i] = batch[0, :, :, :].detach()
            all_labels[i] = labels[0]
            print(all_labels[i], torch.softmax(outputs, 1)[0, 1].item())
            features = model.get_features(batch)[0]
            all_features[i] = features
            i += 1
            if i >= m:
                break
    print(all_predicted == all_labels)
    print(torch.sum(all_predicted == all_labels))
    print()

    # PCA
    features_numpy = all_features.numpy()
    pca = PCA(n_components=16)
    pca_features = pca.fit_transform(features_numpy)
    print("!!!!!!EXPLAINED RATIO: ", np.sum(np.array(pca.explained_variance_ratio_)))

    # dist_prob = calc_all_dist_prob(all_features, template_features)
    dist_prob = calc_all_dist_prob(torch.tensor(pca_features), torch.tensor(pca_features[0]))
    sorted_dist = np.argsort(dist_prob)
    print("Sorted dist: ", sorted_dist)

    thr = 0.6
    for i in range(sq):
        for j in range(sq):
            current_image = cv2.cvtColor(all_images[sorted_dist[i * sq + j]].numpy().transpose((1, 2, 0)) * 255,
                                         cv2.COLOR_RGB2BGR)
            # converted=cv2.cvtColor(current_image,cv2.COLOR_BGR2RGB)
            # result=hist.deconvolution_based_normalization(converted)
            # plt.imshow(converted)
            # plt.show()
            # plt.close()
            # back=cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
            total_image[i * s:i * s + s, j * s:j * s + s] = np.uint8(current_image)

    for i in range(sq):
        for j in range(sq):
            if all_predicted[sorted_dist[i * sq + j]] == 1:
                color = (25, 25, 250)
            else:
                color = (25, 240, 20)
            if all_predicted[sorted_dist[i * sq + j]] != all_labels[sorted_dist[i * sq + j]]:
                color = (0, 0, 0)

            total_image = cv2.putText(total_image, "{:.2f}".format(dist_prob[sorted_dist[i * sq + j]]),
                                      (j * s + 30, i * s + 50),
                                      fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=2,
                                      color=color,
                                      thickness=3,
                                      lineType=cv2.LINE_AA)

    plt.imshow(total_image / 255)
    plt.show()

    cv2.imwrite("balanced_normalized_pca16_L1_5final.png", total_image)
    print("Script worked!")
