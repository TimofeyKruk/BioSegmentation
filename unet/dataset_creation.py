import torch
import os
import cv2
import openslide
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image
from xml.dom import minidom
import matplotlib.pyplot as plt
import torch


class CancerDataset(Dataset):
    def __init__(self,
                 dataset_path="/media/krukts/HDD/BioDiploma/datasetExtracted",
                 # slides_path="/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/wsi",
                 # meta_path="/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/lesion_annotations",
                 ):
        self.dataset_path = dataset_path

        # Iterating over and picking only with non-zero mask
        filenames_list = []

        for folder in os.listdir(dataset_path):
            # print("Folder: ", folder)
            for file in os.listdir(os.path.join(dataset_path, folder)):
                if "_mask" in file:
                    # mask = cv2.imread(os.path.join(dataset_path, folder, file))
                    filenames_list.append(os.path.join(folder, file.split("_mask")[0] + ".png"))

        print("Dataset len: ", len(filenames_list))

        self.filenames_list = filenames_list

    def __len__(self):
        return len(self.filenames_list)

    def __getitem__(self, item):
        filename = self.filenames_list[item]
        # print(filename)
        image = cv2.imread(os.path.join(self.dataset_path, filename))
        mask = cv2.imread(os.path.join(self.dataset_path, filename.split('.')[0] + "_mask.png"))

        # Choosing any channel (it is one dim)
        mask = np.expand_dims(mask[:, :, 0], axis=2)

        # HWC to CHW
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))

        # TODO: Add normalization!!!!
        # Normalizing
        if image.max() > 1:
            image = image / 255
        if mask.max() > 1:
            mask = mask / 255

        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(mask).type(torch.FloatTensor)


if __name__ == '__main__':
    # TODO: THIS CODE TO EXTRACT ONLY USEFUL IMAGES
    dataset_extracted = "/media/krukts/HDD/BioDiploma/datasetExtracted"
    if not os.path.exists(dataset_extracted):
        os.makedirs(dataset_extracted)

    dataset_path = "/media/krukts/HDD/BioDiploma/dataset"

    filenames_list = []

    for folder in os.listdir(dataset_path):
        print("Folder: ", folder)
        for file in os.listdir(os.path.join(dataset_path, folder)):
            if "_mask" in file:
                mask = cv2.imread(os.path.join(dataset_path, folder, file))
                image = cv2.imread(os.path.join(dataset_path, folder, file.split("_mask")[0] + ".png"))

                if mask.any():
                    if not os.path.exists(os.path.join(dataset_extracted, folder)):
                        os.makedirs(os.path.join(dataset_extracted, folder))
                    cv2.imwrite(os.path.join(dataset_extracted, folder, file.split("_mask")[0] + ".png"), image)
                    cv2.imwrite(os.path.join(dataset_extracted, folder, file), mask)
