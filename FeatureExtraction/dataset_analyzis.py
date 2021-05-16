import os
import numpy as np
import csv

if __name__ == '__main__':
    folder = "/media/krukts/HDD/BioDiploma/Timofey/dataset_PINK_300k_Stage2_WithErrorNormCls/Normal"  # TODO: !!!!!
    normal_counter = 0
    tumor_counter = 0
    extra=0
    wsi_set = {}

    print(len(os.listdir(folder)))
    for file in os.listdir(folder):
        wsi_ind = int(file.split("_")[1].split('_')[0])

        cl = None
        if "NRM" in file:
            cl = "NRM"
            normal_counter += 1
        if "TUM" in file:
            cl = "TUM"
            tumor_counter += 1

        if cl is None:
            # print("Not BKG or TUM: ", file)
            extra+=1

        if wsi_ind not in wsi_set:
            wsi_set[wsi_ind] = 1
        else:
            wsi_set[wsi_ind] += 1

    print("len wsi set: ", len(wsi_set))
    print(wsi_set)
    print("TUM: ", tumor_counter)
    print("NRM: ", normal_counter)
    print("extra: ",extra)

    # csv_path=""
    #
    # with open('employee_birthday.txt') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         if line_count == 0:
    #             print(f'Column names are {", ".join(row)}')
    #             line_count += 1
    #         else:
    #             print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
    #             line_count += 1
    #     print(f'Processed {line_count} lines.')
