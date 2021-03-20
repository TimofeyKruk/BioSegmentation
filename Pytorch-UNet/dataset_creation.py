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


class CancerDataset(Dataset):
    def __init__(self,
                 slides_path="/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/wsi",
                 meta_path="/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/lesion_annotations"):
        self.slides_path = slides_path
        self.meta_path = meta_path


if __name__ == '__main__':
    slides_path = "/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/wsi"
    annotations_path = "/media/krukts/HDD/BioDiploma/UIIP_Histo_DATA/CAM17/lesion_annotations"
    # file = "patient_004_node_4.xml"
    save_path = "/media/krukts/HDD/BioDiploma/mini/"

    # mydoc = minidom.parse(os.path.join(annotations_path, file))
    # items = mydoc.getElementsByTagName("Annotation")
    # print('\nAll attributes:')
    # for elem in items:
    #     print(elem.attributes['name'].value)

    level = 5
    size = 512
    for file in os.listdir(slides_path):
        print("FILE: ", file)
        annotation_file = file.split('.')[0] + ".xml"

        if not os.path.exists(os.path.join(save_path, file.split('.')[0])):
            os.makedirs(os.path.join(save_path, file.split('.')[0]))

        tree = ET.parse(os.path.join(annotations_path, annotation_file))
        root = tree.getroot()

        slide = openslide.OpenSlide(os.path.join(slides_path, file))
        x_dim = slide.dimensions[0]
        y_dim = slide.dimensions[1]

        x_level = slide.level_dimensions[level][0]
        y_level = slide.level_dimensions[level][1]
        print(x_level, y_level)
        # tissue_region = slide.get_thumbnail((2500, 2500))
        # scale_x = tissue_region.size[0]
        # scale_y = tissue_region.size[1]
        # tissue_region = np.array(tissue_region)

        # Collecting all contours __________________
        all_contours = []
        for annotation in root[0]:
            print(annotation.attrib["Name"])
            contour = []
            for coordinate in annotation[0]:
                # print("Order= ", float(coordinate.attrib['Order']))
                contour.append(
                    [int(float(coordinate.attrib['X']) / x_dim * x_level),
                     int(float(coordinate.attrib['Y']) / y_dim * y_level)])
            contour = np.array(contour)
            all_contours.append(contour)
            # cv2.fillPoly(tissue_region, pts=[contour], color=(255, 10, 50))

        missed = 0
        # CROPPING images and mask processing ____________
        delta_x = int(size / x_level * x_dim)
        delta_y = int(size / y_level * y_dim)
        for i, x_current in enumerate(range(0, x_dim, delta_x)):
            for j, y_current in enumerate(range(0, y_dim, delta_y)):
                if j == 5 and i == 4:
                    print("j==5")
                cropped_region = slide.read_region((x_current, y_current), level, (size, size))
                cropped_region = cropped_region.convert("RGB")

                cropped_array = np.array(cropped_region)

                mask = np.zeros((size, size))
                # Drawing contours
                for contour in all_contours:
                    post_contour = contour.copy()
                    post_contour[:, 0] -= i * size
                    post_contour[:, 1] -= j * size

                    cv2.fillPoly(mask, pts=[post_contour], color=255)

                    if not (mask == np.zeros((size, size))).all():
                        print("Something drawn!!!")

                if (cropped_array == False).sum() > 0.6 * size * size * 3:
                    missed += 1
                else:
                    cv2.imwrite(os.path.join(save_path, file.split('.')[0], "i{}_j{}.png".format(i, j)), cropped_array)
                    cv2.imwrite(os.path.join(save_path, file.split('.')[0], "i{}_j{}_mask.png".format(i, j)), mask)
                    # Image.fromarray(cropped_array, mode="RGB").save(
                    #     os.path.join(save_path, file.split('.')[0], "i{}_j{}.png".format(i, j)),
                    #     "png")
                    # Image.fromarray(mask, mode="L").save(
                    #     os.path.join(save_path, file.split('.')[0], "i{}_j{}_mask.png".format(i, j)),
                    #     "png")

        print("Missed n: {}".format(missed))

        # # TODO: delete it!!!!
        # print("done")
        # break
        # plt.imshow(tissue_region)
        # plt.show()
        # plt.close()
        # cv2.imwrite(os.path.join(save_path,"{}.jpg".format(file.split('.')[0]), tissue_region))

    # # one specific item attribute
    # print('Item #2 attribute:')
    # print(root[0][1].attrib)
    # (root[0][0][0][2].attrib['X'])
    # # all item attributes
    # print('\nAll attributes:')
    # for elem in root:
    #     for subelem in elem:
    #         print(subelem.attrib)
    #
    # # one specific item's data
    # print('\nItem #2 data:')
    # print(root[0][1].text)
    #
    # # all items data
    # print('\nAll item data:')
    # for elem in root:
    #     for subelem in elem:
    #         print(subelem.text)
    print("Scipt worked!")
