import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from dataset_creation import CancerDataset
from torch.utils.data import DataLoader, random_split
import cv2
import matplotlib.pyplot as plt


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    # parser.add_argument('--scale', '-s', type=float,
    #                     help="Scale factor for the input images",
    #                     default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model_path = "checkpoints4/CP_epoch20.pth"

    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(model_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))

    logging.info("Model loaded !")

    dataset = CancerDataset()
    print("DATASET LOADED!!!")

    val_percent = 0
    batch_size = 1

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    i = 0
    with torch.no_grad():
        for imgs, true_masks in train_loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)

            predicted_masks = net(imgs)
            probabilities = torch.sigmoid(predicted_masks)

            print(probabilities.shape)
            print("I: ", i)
            prob = (probabilities.cpu() * 255)[0].numpy().transpose((1, 2, 0))
            print(prob.shape)

            threshold = 0.5
            prob[prob < threshold * 255] = 0

            cv2.imwrite("rubbish4/{}_predicted.png".format(i),
                        cv2.applyColorMap(prob.astype(np.uint8), cv2.COLORMAP_JET))
            cv2.imwrite("rubbish4/{}_image.png".format(i), (imgs.cpu() * 255)[0].numpy().transpose((1, 2, 0)))
            cv2.imwrite("rubbish4/{}_mask_original.png".format(i),
                        (true_masks.cpu() * 255)[0].numpy().transpose((1, 2, 0)))
            # plt.imshow(prob)
            # plt.show()
            # plt.close()
            i += 1
            if i >= 80:
                break
