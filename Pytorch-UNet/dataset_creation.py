import torch
import os
import cv2
import openslide
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
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
    file = "patient_004_node_4.xml"

    # mydoc = minidom.parse(os.path.join(annotations_path, file))
    # items = mydoc.getElementsByTagName("Annotation")
    # print('\nAll attributes:')
    # for elem in items:
    #     print(elem.attributes['name'].value)

    for file in os.listdir(slides_path):
        print("FILE: ", file)
        annotation_file = file.split('.')[0] + ".xml"

        level
        slide = openslide.OpenSlide(os.path.join(slides_path, file))
        x_dim = slide.dimensions[0]
        y_dim = slide.dimensions[1]
        tissue_region = slide.get_thumbnail((2500, 2500))
        scale_x = tissue_region.size[0]
        scale_y = tissue_region.size[1]
        tissue_region = np.array(tissue_region)

        tree = ET.parse(os.path.join(annotations_path, annotation_file))
        root = tree.getroot()

        for annotation in root[0]:
            print(annotation.attrib["Name"])
            contour = []
            for coordinate in annotation[0]:
                print("Order= ", float(coordinate.attrib['Order']))
                contour.append(
                    [int(float(coordinate.attrib['X']) / x_dim * scale_x),
                     int(float(coordinate.attrib['Y']) / y_dim * scale_y)])
            contour = np.array(contour)

            cv2.fillPoly(tissue_region, pts=[contour], color=(255, 10, 50))


        # plt.imshow(tissue_region)
        # plt.show()
        # plt.close()
        cv2.imwrite("/media/krukts/HDD/BioDiploma/mini/{}.jpg".format(file.split('.')[0]), tissue_region)

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
