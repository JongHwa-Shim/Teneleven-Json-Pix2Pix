import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_numpy, filename):
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

color_map = {'background': [0, 0, 0], 'entrance': [0, 66, 0], 'utility': [0, 0, 66],
            'dress': [125, 0, 0], 'toilet': [0, 125, 0], 'balcony':[0, 0, 125],
            'bed': [255, 0, 0], 'kitchen': [0, 255, 0], 'living': [0, 0, 255],
            'window': [66, 66, 0], 'slide': [66, 0, 66], 'door': [0, 66, 66],
            'enter': [125, 125, 0], 'wall': [125, 0, 125], 'extra': [0, 125, 125], 'dinning': [255, 0, 0], 'silhouette': [125, 125, 125]}

room_list_target = {0: 'background', 1: 'entrance', 2: 'utility', 3: 'dress', 4: 'toilet', 
                5: 'balcony', 6: 'bed', 7: 'dinning', 8: 'kitchen', 9: 'living', 
                10: 'window', 11: 'slide', 12: 'door', 13: 'enter', 14: 'wall', 15: 'extra'}

def plan_2_img_json(img_tensor):
    """
    Args:
        img_tensor (torch.tensor): (1 x C x H x W)
    
    Returns:
        canvas (np.array): (H x W x 3)
    """

    img = torch.argmax(img_tensor, dim=1) # (1 x H x W)
    img = np.array(img)
    img = np.squeeze(img, axis=0) # (H x w)
    canvas = np.zeros([img.shape[0], img.shape[1], 3]) # (H x W x 3)

    for room_num, room_name in room_list_target.items():
        color = color_map[room_name]
        color_mask = np.where(img == room_num, True, False)
        canvas[color_mask] = color

    return canvas

# room_list_input = {0: 'silhouette', 1: 'entrance', 2: 'utility', 3: 'dress', 4: 'toilet', 
#                 5: 'balcony', 6: 'bed', 7: 'dinning', 8: 'kitchen', 9: 'living', 
#                 10: 'window', 11: 'slide', 12: 'door', 13: 'enter', 14: 'extra'}

# exp6 changes
room_list_input = {0: 'silhouette', 1: 'window', 2: 'enter', 3: 'entrance', 4: 'living'}

import torch
def coloring_plan_json(img_tensor):
    """
    Convert random instanced image to color image.

    Args:
        img_tensor (Tensor): (1 x C x H x w)
    
    Returns:
        canvas (np.array): (H x W x 3)
    """

    img_tensor = torch.squeeze(img_tensor, dim=0) # (C x H x W)
    canvas = np.zeros((img_tensor.shape[1], img_tensor.shape[2], 3)) # (H x W x 3)
    cha_num = img_tensor.shape[0]
    
    # draw silhouette first
    sil_mask = np.array(torch.where(img_tensor[0] == 1, True, False).cpu())
    canvas[sil_mask] = color_map[room_list_input[0]]

    for cha in range(1, cha_num):
        color_mask = np.array(torch.where(img_tensor[cha] == 1, True, False).cpu())
        canvas[color_mask] = color_map[room_list_input[cha]]
    
    print(np.unique(canvas))
    return canvas