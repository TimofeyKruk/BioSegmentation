import torch
import os
import cv2
from FeatureExtraction.model import FEModel
from FeatureExtraction.dataset import CAM17Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from torch.utils.data import DataLoader
from torchsummary import summary
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN

shuffle_dataset = True
save_path = "/media/krukts/HDD/BioDiploma/FeatureExtraction/log"
save_name = "efficientnet-b0after12"

dataset_path = "/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls"
img_size = (224, 224)
tb_name = "FirstTryPOST_Kmeans"

if __name__ == '__main__':
    model = FEModel()
    model.load_state_dict(torch.load(os.path.join(save_path, save_name)))
    print("Model was loaded: ", save_name)

    model.to("cpu")
    print(model)
    print(model.parameters())
    print("$$$$$$$$$$$$$$$$")
    print("Torchsummary: _______________")
    summary(model, [(3, 224, 224)], batch_size=16, device="cpu")

    criterion = torch.nn.CrossEntropyLoss()

    # Dataset creation
    dataset = CAM17Dataset(dataset_path=dataset_path, img_size=img_size)
    train_dataloader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=True,
                                  drop_last=True)

    writer = SummaryWriter("/media/krukts/HDD/BioDiploma/FeatureExtraction/log/tb/" + tb_name)

    i = 0
    m = 400

    all_features = torch.zeros((m, 256))
    all_images = torch.ones((m, 3, 224, 224))
    all_labels = torch.ones(m)

    with torch.no_grad():
        for batch, labels in train_dataloader:
            print(i)
            outputs = model(batch)

            all_images[i] = batch[0, :, :, :].detach()
            all_labels[i] = labels[0]

            features = model.get_features(batch)[0]
            all_features[i] = features
            i += 1
            if i >= m:
                break


    k_means = KMeans(n_clusters=2, random_state=1)
    k_means.fit(all_features.numpy())
    clusters, centers = k_means.labels_, k_means.cluster_centers_

    writer.add_embedding(all_features,
                         metadata=torch.tensor(clusters),
                         label_img=all_images)
    writer.close()
    print("Script worked!")