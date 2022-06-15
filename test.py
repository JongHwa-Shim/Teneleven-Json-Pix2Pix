from __future__ import print_function
import argparse
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import *
from dataset import *

# Testing settings
def test(epoch):
    parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
    parser.add_argument('--dataset', required=True, help='facades')
    parser.add_argument('--nepochs', type=int, default=100, help='saved model of which epochs')
    parser.add_argument('--cuda', action='store_true', help='use cuda')
    opt = parser.parse_args("--dataset json --nepochs {} --cuda".format(epoch).split())
    print(opt)

    device = torch.device("cuda:1" if opt.cuda else "cpu")

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
    for img_input, img_real, dir in testing_data_loader:
        # generate images
        img_input = img_input.to(device)
        out = net_g(img_input)
        out_img = out.detach().cpu()

        # colorize output image
        out_img = plan_2_img_json(out_img) # numpy (H, W, 3)
        
        # deny output image pixels out of silhouette
        inv_sil_mask = np.where(img_input.cpu()[0,0] == 1, False, True) # numpy (H, W)
        out_img[inv_sil_mask] = [0,0,0]

        # colorize input image
        img_input = coloring_plan_json(img_input) # numpy (H, W, 3)

        # colorize real image
        out_img_real = plan_2_img_json(img_real)

        # make figure
        final_img = np.concatenate([img_input, out_img, out_img_real], axis=1)

        # make save directory and save figures
        if not os.path.exists(os.path.join("./result", opt.dataset + str(opt.nepochs))):
            os.makedirs(os.path.join("./result", opt.dataset + str(opt.nepochs)))
        save_img(final_img, "result/{}/{}.png".format(opt.dataset + str(opt.nepochs), dir))
        

if __name__ == '__main__':
    start = 1
    end = 3
    step = 1
    for i in range(start, end, step):
        test(i)