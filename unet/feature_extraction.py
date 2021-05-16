import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans, DBSCAN

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from dataset_creation import CancerDataset
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model_path = "checkpoints2/CP_epoch20.pth"

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    logging.info("Model loaded !")

    dataset = CancerDataset()
    print("DATASET LOADED!!!")

    val_percent = 0
    batch_size = 1

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    tb_name="Try3"
    writer = SummaryWriter("/media/krukts/HDD/BioDiploma/unet/rubbish5simple/tensorboard/"+tb_name)

    i = 0
    m = 330

    all_features = torch.zeros((m, 512))
    all_images = torch.ones((m, 3, 64, 64))

    with torch.no_grad():
        for imgs, true_masks in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            predicted_masks = net(imgs)
            probabilities = torch.sigmoid(predicted_masks)

            # print(probabilities.shape)
            print("I: ", i)
            prob = (probabilities.cpu() * 255)[0].numpy().transpose((1, 2, 0))
            # print(prob.shape)

            threshold = 0.7
            prob[prob < threshold * 255] = 0

            cv2.imwrite("rubbish5simple/{}_predicted.png".format(i),
                        cv2.applyColorMap(prob.astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imwrite("rubbish5simple/{}_image.png".format(i), (imgs.cpu() * 255)[0].numpy().transpose((1, 2, 0)))
            cv2.imwrite("rubbish5simple/{}_mask_original.png".format(i),
                        (true_masks.cpu() * 255)[0].numpy().transpose((1, 2, 0)))

            features = net.predict(imgs)
            # print("Features.shape: ", features.shape)

            # all_features.append(features[0, :].cpu())
            # all_images.append(imgs[0, :, :, :].cpu().transpose(0, 2))
            all_images[i] = transforms.Scale((64, 64))(imgs[0, :, :, :].detach().cpu())
            all_features[i] = features[0, :].detach().cpu()

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
