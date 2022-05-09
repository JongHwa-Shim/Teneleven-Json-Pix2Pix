from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import is_image_file, load_img, plan_2_img, save_img, coloring_plan
from dataset import JsonDatasetTest, RPLANDataset_test

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args("--dataset json --nepochs 1 --cuda".split())
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

model_path = "checkpoint/{}/netG_model_epoch_{}.pth".format(opt.dataset, opt.nepochs)

net_g = torch.load(model_path).to(device)

"""
if opt.direction == "a2b":
    image_dir = "dataset/{}/test/a/".format(opt.dataset)
else:
    image_dir = "dataset/{}/test/b/".format(opt.dataset)

image_filenames = [x for x in os.listdir(image_dir) if is_image_file(x)]

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

for image_name in image_filenames:
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join("result", opt.dataset)):
        os.makedirs(os.path.join("result", opt.dataset))
    save_img(out_img, "result/{}/{}".format(opt.dataset, image_name))
"""

test_path = "dataset/{}/test".format(opt.dataset)

test_set = JsonDatasetTest(test_path)
testing_data_loader = DataLoader(dataset=test_set,  batch_size=1, shuffle=False)

import numpy as np
for img, dir in testing_data_loader:
    img = img.to(device)
    out = net_g(img)
    out_img = out.detach().cpu()
    if not os.path.exists(os.path.join("./result", opt.dataset)):
        os.makedirs(os.path.join("./result", opt.dataset))
    out_img = plan_2_img(out_img) # numpy (H, W, 3)
    img = coloring_plan(img) # numpy (H, W, 3)
    final_img = np.concatenate([img, out_img], axis=1)
    save_img(final_img, "result/{}/{}.png".format(opt.dataset, dir))

