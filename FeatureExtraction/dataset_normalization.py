import os
import cv2
import numpy as np
import histomicstk.preprocessing.color_normalization as hist

if __name__ == '__main__':
    dataset_path = "/media/krukts/HDD/BioDiploma/BalancedFEData/HISTO_Image_Dataset_CAMELYON16_3_Classes_18K_Tiles"

    folder = "Class_3b_EPI_Test"
    new_folder = folder + "_NORMALIZED"
    if not os.path.exists(os.path.join(dataset_path, new_folder)):
        os.makedirs(os.path.join(dataset_path, new_folder))

    for file in os.listdir(os.path.join(dataset_path, folder)):
        print(file)
        img = cv2.imread(os.path.join(dataset_path, folder, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_normalized = hist.deconvolution_based_normalization(img)
        img_back = cv2.cvtColor(img_normalized, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(dataset_path, new_folder, file), img_back)
