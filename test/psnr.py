import math
import cv2
import numpy as np
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="Calc psnr")

    parser.add_argument("--input", type=str, default="", help="Image to calculate psnr.")
    parser.add_argument("--ref", type=str, default="", help="Reference image.")
    return parser.parse_args()


def PSNR(original, compressed):
    mse = np.mean((original - compressed)**2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr


if __name__ == "__main__":
    args = parse_args()
    original = cv2.imread(args.ref)
    input_path = args.input
    for name in os.listdir(input_path):
        if name.split(".")[-1] == "jpg":
            input_img = cv2.imread(os.path.join(input_path, name))
            value = PSNR(original, input_img)
            print("PSNR value is {:.3f} dB for img {}".format(value, name))
