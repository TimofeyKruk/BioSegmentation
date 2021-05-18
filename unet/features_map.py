import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
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


def calc_all_dist_prob(examples, anchor):
    dist = np.zeros(examples.shape[0])
    for i in range(examples.shape[0]):
        dist[i] = torch.sqrt(torch.sum((anchor - examples[i]) ** 2)).item()

    # dist /= np.max(dist)
    # dist = 1 - dist
    return dist


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model_path = "checkpoints4/CP_epoch20.pth"

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

    tb_name = "Try2_BW"
    writer = SummaryWriter("/media/krukts/HDD/BioDiploma/unet/rubbish5simple/tensorboard/" + tb_name)

    i = 0
    m = 36
    s = 224
    alpha = 0.4

    all_features = torch.zeros((m, 1024))
    all_images = torch.ones((m, 3, s, s))

    all_predicted = np.zeros((m, s, s))

    sq = int(math.sqrt(m))
    total_masked_image = np.zeros((sq * s, sq * s, 3))
    total_color = np.zeros((sq * s, sq * s, 3))
    template_features = None

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

            threshold = 0.5
            prob[prob < threshold * 255] = 0

            # TODO: Extracting features
            features = net.predict(imgs)
            if i == 0:
                template_features = features.detach().cpu()

            # print("Features.shape: ", features.shape)

            # all_features.append(features[0, :].cpu())
            # all_images.append(imgs[0, :, :, :].cpu().transpose(0, 2))

            all_images[i] = transforms.Resize((s, s))(imgs[0, :, :, :].detach().cpu())
            all_predicted[i] = cv2.resize(prob, (s, s))
            all_features[i] = features[0, :].detach().cpu()

            i += 1
            if i >= m:
                break

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
            current_image = all_images[sorted_dist[i * sq + j]].numpy().transpose((1, 2, 0)) * 255
            current_predicted = all_predicted[sorted_dist[i * sq + j]]

            total_masked_image[ j * s:j * s + s,i * s:i * s + s] = np.uint8((1 - alpha) * current_image)

            # current_heatmap = np.full((s, s, 3), 255.)
            # current_heatmap *= dist_prob[i * sq + j] / np.max(dist_prob)

            # if dist_prob[i * sq + j] >= thr:
            #     current_heatmap[:, :, [0, 2]] = 0
            # else:
            #     current_heatmap[:, :, 2] = 0
            current_heatmap = cv2.applyColorMap(np.uint8(current_predicted), cv2.COLORMAP_HOT)
            current_heatmap[:, :, 2] = 0

            total_masked_image[ j * s:j * s + s,i * s:i * s + s,] += np.uint8(alpha * current_heatmap)

    for i in range(sq):
        for j in range(sq):
            total_masked_image = cv2.putText(total_masked_image, "{:.2f}".format(dist_prob[sorted_dist[i * sq + j]]),
                                             (j * s + 30, i * s + 50),
                                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                             fontScale=1.5,
                                             color=(25, 5, 10),
                                             thickness=2,
                                             lineType=cv2.LINE_AA)

    plt.imshow(total_masked_image / 255)
    plt.show()

    cv2.imwrite("out_pca8new.png", total_masked_image)
    print("Script worked!")
