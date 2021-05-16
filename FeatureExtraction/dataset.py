import os
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms


class CAM17Dataset(Dataset):
    def __init__(self, dataset_path="/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls",
                 img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size

        self.all_names = []
        self.all_labels = []

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        # TODO: REDO DATASET FULLY, as discussed with VA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Adding NORMAL
        for file in os.listdir(os.path.join(self.dataset_path, "Normal")):
            # wsi_ind = int(file.split("_")[1].split('_')[0])
            cl = None
            if "NRM" in file:
                cl = "NRM"
                self.all_labels.append(0)
            if "TUM" in file:
                cl = "TUM"
                self.all_labels.append(1)

            if cl is not None:
                self.all_names.append(file)
        # TODO: DELETE [:50000]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for file in os.listdir(os.path.join(self.dataset_path, "Tumor"))[:50000]:
            # wsi_ind = int(file.split("_")[1].split('_')[0])
            cl = None
            if "NRM" in file:
                cl = "NRM"
                self.all_labels.append(0)
            if "TUM" in file:
                cl = "TUM"
                self.all_labels.append(1)

            if cl is not None:
                self.all_names.append(file)

        print("Len dataset: ", len(self))

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, index):
        label = self.all_labels[index]
        subfolder = "Tumor" if label else "Normal"
        image = cv2.imread(os.path.join(self.dataset_path, subfolder, self.all_names[index]))

        # TODO: IMPORTANT!!! Resizing
        image = cv2.resize(image, self.img_size)
        image = self.transform(image)

        return image, torch.tensor(label)
