import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    image=cv2.imread("/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls/Normal/Normal_001_tilec_BKG_63.png")

    plt.imshow(image)
    plt.show()

    im2=transform(image)

    print(im2)
