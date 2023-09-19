import cv2
import torch
import time
import os
import shutil
import numpy as np
from multiprocessing import Process
import argparse

def image_resize(image, scale,interpolation):
    im = cv2.imread(image)
    r = im.shape[0] / im.shape[1]
    new_dim = (int(im.shape[1] * scale), int(im.shape[0] * scale))
    resized = cv2.resize(im, new_dim, interpolation=interpolation)
    return resized


def dataset_resize(old_dir, new_dir, scale):
    try:
        os.makedirs(new_dir)
    except FileExistsError:
        pass
    for root, dirs, files in os.walk(old_dir):
        print(root)
        print(root.replace(old_dir, new_dir, 1))
        for name in files:
            if name.endswith(".jpg"):
                resized = image_resize(f"{root}/{name}", scale,interpolation=cv2.INTER_AREA)
                cv2.imwrite(root.replace(old_dir, new_dir, 1) + f"/{name}", resized)
            elif name.endswith(".png"):
                resized = image_resize(f"{root}/{name}", scale,interpolation=cv2.INTER_NEAREST_EXACT)
                cv2.imwrite(root.replace(old_dir, new_dir, 1) + f"/{name}", resized)
            elif name.endswith(".npz"):
                continue
            else:
                shutil.copyfile(f"{root}/{name}", root.replace(old_dir, new_dir, 1) + f"/{name}")
        print(dirs)
        for dir in dirs:
            print(dir)
            print(root.replace(old_dir, new_dir, 1))
            try:
                os.makedirs(root.replace(old_dir, new_dir, 1) + "/" + dir)
            except FileExistsError:
                continue




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root-dir", default=None)
    parser.add_argument("--new-root-dir", default=None)
    opt = parser.parse_args()
    if opt.old_root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.old_root_dir) or not os.path.isdir(opt.old_root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)
    if opt.new_root_dir is None:
        print("pls need dataset root directory, fuckuuuuuu")
        exit(-1)
    if not os.path.exists(opt.new_root_dir) or not os.path.isdir(opt.new_root_dir):
        print("pls need actual existing dataset root directory, fuckuuuuuu")
        exit(-1)

    p1 = Process(target=dataset_resize,

                 args=(os.path.join(opt.old_root_dir,"dresses"),os.path.join(opt.new_root_dir,"dresses"), 0.5), daemon=True)
    p2 = Process(target=dataset_resize,
                 args=(os.path.join(opt.old_root_dir,"upper_body"),os.path.join(opt.new_root_dir,"upper_body"), 0.5), daemon=True)
    p3 = Process(target=dataset_resize,
                 args=(os.path.join(opt.old_root_dir,"lower_body"),os.path.join(opt.new_root_dir,"lower_body"), 0.5), daemon=True)

    p1.start(), p2.start(), p3.start()
    p1.join()
    p2.join()
    p3.join()
