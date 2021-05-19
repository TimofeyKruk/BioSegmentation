import os
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

shuffle_dataset = True
save_path = "/media/krukts/HDD/BioDiploma/FeatureExtraction/log"
dataset_path = "/media/krukts/HDD/BioDiploma/BalancedFEData/HISTO_Image_Dataset_CAMELYON16_3_Classes_18K_Tiles"
img_size = (224, 224)

# What weights to load. Specify here
save_name = "BalancedNormalized"

features_file = "Test_3classes_{}.json".format(save_name)

if __name__ == '__main__':
    # This script runs and save features of specific folder
    model = FEModel()
    model.load_state_dict(torch.load(os.path.join(save_path, save_name)))
    print("Model was loaded: ", save_name)

    # if torch.cuda.is_available():
    #     device="gpu"
    # else:
    #     device="cpu"
    device = "cpu"
    model.to(device)

    # Dataset creation
    dataset = CAM17Dataset(type="test", dataset_path=dataset_path, img_size=img_size, extended=True,
                           includeEPI=True) #check this includeEPI (depending on what you need)
    train_dataloader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=1,
                                  shuffle=False)

    i = 0

    all_data = {}

    model.eval()
    with torch.no_grad():
        for batch, labels, wsi, filename in train_dataloader:
            print(i)
            outputs = model(batch)

            image = batch[0, :, :, :].detach()
            label = labels[0].item()
            wsi_index = [wsi[0][0], wsi[1].item()]
            file = filename[0]
            features = model.get_features(batch)[0].numpy()

            # TODO: WON'T work for 3 classes
            predicted = int(torch.softmax(outputs, 1)[0, 1].item() > 0.5)

            all_data[file] = {"label": label,
                              "predicted": predicted,
                              "wsi_index": wsi_index,
                              "features": [float(f) for f in features]}
            i += 1


    with open(os.path.join(dataset_path, features_file), 'w') as file_out:
        json.dump(all_data, file_out, indent=4)
    print("Script worked!")
