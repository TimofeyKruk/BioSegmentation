import cv2
import numpy as np
import os

from dataset_creation import CancerDataset

if __name__ == '__main__':
    dataset = CancerDataset()
    print("LEN DATASET: ", len(dataset))
