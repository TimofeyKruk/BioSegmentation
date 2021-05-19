import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import histomicstk.preprocessing.color_normalization as hist

if __name__ == '__main__':
    pink_path = "/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls/Tumor/Tumor_001_tilec_TUM2_1.png"
    blue_path = "/media/krukts/HDD/BioDiploma/Timofey/dataset_BLUE_300k/Tumor/tumor_071_tilec_TUM2_15.png"

    pink = cv2.imread(pink_path)
    pink = cv2.cvtColor(pink, cv2.COLOR_BGR2RGB)
    blue = cv2.imread(blue_path)
    blue = cv2.cvtColor(blue, cv2.COLOR_BGR2RGB)

    plt.title("PINK")
    plt.imshow(pink)
    plt.show()
    plt.close()

    plt.title("BLUE")
    plt.imshow(blue)
    plt.show()
    plt.close()
    # cv2.imwrite("l.png",pink)

    # Normalization
    result_pink = hist.deconvolution_based_normalization(pink)
    result_blue = hist.deconvolution_based_normalization(blue)

    plt.title("PostPink")
    plt.imshow(result_pink)
    plt.show()
    plt.close()

    plt.title("PostBlue")
    plt.imshow(result_blue)
    plt.show()
    plt.close()
    # cv2.imwrite("pink.png",pink)
    # cv2.imwrite("resultpink.png",result_pink)