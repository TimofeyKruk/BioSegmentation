import os
from torch.utils.data import Dataset
import cv2
import torch
from torchvision import transforms


class CAM17Dataset(Dataset):
    def __init__(self,
                 type="train",
                 dataset_path="/media/krukts/HDD/BioDiploma/BalancedFEData/HISTO_Image_Dataset_CAMELYON16_3_Classes_18K_Tiles",
                 img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.type = type

        self.all_names = []
        self.all_labels = []
        self.all_wsi_indexes = []

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        nrm_subfolder = None
        tum_subfolder = None
        if self.type == "train":
            nrm_subfolder = "Class_1a_NRM_Train"
            tum_subfolder = "Class_2a_TUM_Train"
        if self.type == "test":
            nrm_subfolder = "Class_1b_NRM_Test"
            tum_subfolder = "Class_2b_TUM_Test"

        self.nrm_subfolder = nrm_subfolder
        self.tum_subfolder = tum_subfolder

        # Adding NRM
        for file in os.listdir(os.path.join(self.dataset_path, self.nrm_subfolder)):
            split = file.split('_')

            assert split[-2] == "NRM"
            self.all_labels.append(0)
            self.all_wsi_indexes.append((split[0], int(split[1])))
            # print("NRM : ", (split[0], int(split[1])))

            self.all_names.append(os.path.join(self.nrm_subfolder, file))

        # Adding TUM
        for file in os.listdir(os.path.join(self.dataset_path, self.tum_subfolder)):
            split = file.split('_')

            assert split[-2] == "TUM2"
            self.all_labels.append(1)
            self.all_wsi_indexes.append((split[0], int(split[1])))
            # print("TUM : ", (split[0], int(split[1])))

            self.all_names.append(os.path.join(self.tum_subfolder, file))

        print("Len dataset: ", len(self))

    def __len__(self):
        return len(self.all_names)

    def __getitem__(self, index):
        label = self.all_labels[index]

        image = cv2.imread(os.path.join(self.dataset_path, self.all_names[index]))

        # TODO: IMPORTANT!!! Resizing
        image = cv2.resize(image, self.img_size)
        image = self.transform(image)

        return image, torch.tensor(label)

# # OLD
# class CAM17Dataset(Dataset):
#     def __init__(self, dataset_path="/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls",
#                  img_size=(224, 224)):
#         self.dataset_path = dataset_path
#         self.img_size = img_size
#
#         self.all_names = []
#         self.all_labels = []
#
#         self.transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         # TODO: REDO DATASET FULLY, as discussed with VA!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         # Adding NORMAL
#         for file in os.listdir(os.path.join(self.dataset_path, "Normal")):
#             # wsi_ind = int(file.split("_")[1].split('_')[0])
#             cl = None
#             if "NRM" in file:
#                 cl = "NRM"
#                 self.all_labels.append(0)
#             if "TUM" in file:
#                 cl = "TUM"
#                 self.all_labels.append(1)
#
#             if cl is not None:
#                 self.all_names.append(file)
#         # TODO: DELETE [:50000]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#         for file in os.listdir(os.path.join(self.dataset_path, "Tumor"))[:50000]:
#             # wsi_ind = int(file.split("_")[1].split('_')[0])
#             cl = None
#             if "NRM" in file:
#                 cl = "NRM"
#                 self.all_labels.append(0)
#             if "TUM" in file:
#                 cl = "TUM"
#                 self.all_labels.append(1)
#
#             if cl is not None:
#                 self.all_names.append(file)
#
#         print("Len dataset: ", len(self))
#
#     def __len__(self):
#         return len(self.all_names)
#
#     def __getitem__(self, index):
#         label = self.all_labels[index]
#         subfolder = "Tumor" if label else "Normal"
#         image = cv2.imread(os.path.join(self.dataset_path, subfolder, self.all_names[index]))
#
#         # TODO: IMPORTANT!!! Resizing
#         image = cv2.resize(image, self.img_size)
#         image = self.transform(image)
#
#         return image, torch.tensor(label)
