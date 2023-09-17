import argparse


def get_mask_from_labels(img):
    img = img[img!=0] = 255


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--label-dir",default=None)
    parser.add_argument("--mask-dir", default=None)
    opt = parser.parse_args()