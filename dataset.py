from os import listdir
from os.path import join
import random
import os
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils import is_image_file, load_img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, direction):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = join(image_dir, "a")
        self.b_path = join(image_dir, "b")
        self.image_filenames = [x for x in listdir(self.a_path) if is_image_file(x)]

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        a = a.resize((286, 286), Image.BICUBIC)
        b = b.resize((286, 286), Image.BICUBIC)
        a = transforms.ToTensor()(a)
        b = transforms.ToTensor()(b)
        w_offset = random.randint(0, max(0, 286 - 256 - 1))
        h_offset = random.randint(0, max(0, 286 - 256 - 1))
    
        a = a[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
        b = b[:, h_offset:h_offset + 256, w_offset:w_offset + 256]
    
        a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        if random.random() < 0.5:
            idx = [i for i in range(a.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            a = a.index_select(2, idx)
            b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)

class RPLANDataset(data.Dataset):
    def __init__(self, image_root, transform=None):
        super(RPLANDataset, self).__init__()

        color_dict = {'Living room': [255, 0, 0], 'Bed room': [0, 255, 0], 'Kitchen': [0, 0, 255], 'Bathroom': [255, 255, 0], 'Balcony': [255, 0, 255],
                    'Entrance': [0, 255, 255], 'Extra room': [125, 255, 255], 'Front door': [255, 255, 255], 'Door': [255, 125, 125], 'Wall': [125, 125, 125], 'Outside': [0, 0, 0]}
        self.room_list = ['silouhette', 'Living room', 'Bed room', 'Kitchen', 'Bathroom', 'Balcony', 'Entrance', 'Extra room', 'Front door', 'Door', 'Wall']
        self.room_num = {'silouhette': 0, 'Living room': 1, 'Bed room': 2, 'Kitchen': 3, 'Bathroom': 4, 'Balcony': 5, 'Entrance': 6, 'Extra room': 7, 'Front door': 8, 'Door': 9, 'Wall': 10}

        self.img_root = image_root

        self.room_dir_list = [img_dir for img_dir in listdir(self.img_root)]

        self.transform = transform

    def __getitem__(self, index):
        room_path = join(self.img_root, self.room_dir_list[index])
        exist_room_list_ext = listdir(room_path)
        exist_room_list = [exist_room.split('.')[0] for exist_room in exist_room_list_ext]

        room_dict = {}
        for room in self.room_list:
            if room in exist_room_list:
                room_img_pth = join(room_path, room + '.png')
                room_img = Image.open(room_img_pth)
                room_dict[room] = room_img
            else:
                room_dict[room] = None

        room_tensor_list = [0] * len(self.room_num)
        room_tensor_list_input = [0] * len(self.room_num)
        for room_name, room_img in room_dict.items():
            if room_img == None:
                room_tensor_list[self.room_num[room_name]] = torch.zeros([1, 256, 256])
                room_tensor_list_input[self.room_num[room_name]] = torch.zeros([1, 256, 256])
            else:
                room_img = transforms.PILToTensor()(room_img)

                # using random instance
                if self.room_num[room_name] == 0:
                    room_tensor_list_input[0] = torch.where(room_img != 0, 1, 0)
                else:
                    room_tensor_list_input[self.room_num[room_name]] = random_instance(room_img)


                room_img = torch.where(room_img != 0, 1, 0)
                room_tensor_list[self.room_num[room_name]] = room_img
        
        input = torch.cat(room_tensor_list_input[0:10], dim=0)
        target = torch.cat(room_tensor_list[1:11], dim=0)

        if self.transform == None:
            pass
        else:
            input, target = self.transform(input, target)
        return input, target

    def __len__(self):
        return len(self.room_dir_list)
    
class RPLANDataset_test(data.Dataset):
    def __init__(self, image_root):
        super(RPLANDataset_test, self).__init__()

        color_dict = {'Living room': [255, 0, 0], 'Bed room': [0, 255, 0], 'Kitchen': [0, 0, 255], 'Bathroom': [255, 255, 0], 'Balcony': [255, 0, 255],
                    'Entrance': [0, 255, 255], 'Extra room': [125, 255, 255], 'Front door': [255, 255, 255], 'Door': [255, 125, 125], 'Wall': [125, 125, 125], 'Outside': [0, 0, 0]}
        self.room_list = ['silouhette', 'Living room', 'Bed room', 'Kitchen', 'Bathroom', 'Balcony', 'Entrance', 'Extra room', 'Front door', 'Door', 'Wall']
        self.room_num = {'silouhette': 0, 'Living room': 1, 'Bed room': 2, 'Kitchen': 3, 'Bathroom': 4, 'Balcony': 5, 'Entrance': 6, 'Extra room': 7, 'Front door': 8, 'Door': 9, 'Wall': 10}

        self.plan_img_list = []
        for image_dir in listdir(image_root):

            room_path = join(image_root, image_dir)
            exist_room_list_ext = listdir(room_path)
            exist_room_list = [exist_room.split('.')[0] for exist_room in exist_room_list_ext]

            room_dict = {}
            for room in self.room_list:
                if room in exist_room_list:
                    room_img_pth = join(room_path, room + '.png')
                    room_img = Image.open(room_img_pth)
                    room_dict[room] = room_img
                else:
                    room_dict[room] = None
            room_dict['dir'] = image_dir
            
            self.plan_img_list.append(room_dict)

    def __getitem__(self, index):
        room_dict = self.plan_img_list[index]
        
        room_tensor_list = [0] * (len(room_dict) - 1) # exclude 'dir'
        for room_name in self.room_list:

            if room_dict[room_name] == None:
                room_tensor_list[self.room_num[room_name]] = torch.zeros((1, 256, 256))
            else:
                room_img = transforms.PILToTensor()(room_dict[room_name])

                if self.room_num[room_name] == 0: # silouhette
                    room_tensor_list[self.room_num[room_name]] = torch.where(room_img != 0, 1, 0)
                else:
                    room_tensor_list[self.room_num[room_name]] = random_instance(room_img)

        input = torch.cat(room_tensor_list, dim=0).to(torch.float32)
        room_dir = room_dict['dir']
        
        return input, room_dir

    def __len__(self):
        return len(self.plan_img_list)

import random
import numpy as np
def random_instance(room_img: torch.tensor):
    """
    Implement instance randomly.

    Args:
        room_img (Tensor): (1 x 256 x 256)
    
    Returns:
        img (Tensor): (1 x 256 x 256) random instance implemented image
    """

    # delete background
    ins_num_list = torch.unique(room_img)
    ins_num_list = ins_num_list[ins_num_list!=0]

    # select number of implementing instance 
    num_ins = len(ins_num_list)
    num_ins = random.randrange(0, num_ins+1)
    
    # choose instance
    ins_num_list = random.sample(list(ins_num_list), num_ins)

    # implement choosed instance
    ins_canvas = torch.zeros_like(room_img)
    for ins_num in ins_num_list:
        ins_mask = torch.where(room_img == ins_num, True, False)
        ins_canvas[ins_mask] = 1
    
    return ins_canvas

# room_list_input = {'entrance': 1, 'utility': 2, 'dress': 3, 'toilet': 4, 'balcony': 5, 'bed': 6, 'dinning': 7, 'kitchen': 8, 'living': 9, 'window': 10,
#             'slide': 11, 'door': 12, 'enter': 13, 'extra': 14}
room_list_input = {'entrance': 1, 'window': 10, 'door': 12, 'enter': 13}

room_list_target = {'entrance': 1, 'utility': 2, 'dress': 3, 'toilet': 4, 'balcony': 5, 'bed': 6, 'dinning': 7, 'kitchen': 8, 'living': 9, 'window': 10,
            'slide': 11, 'door': 12, 'enter': 13, 'wall': 14, 'extra': 15}
class JsonDatasetTrain(data.Dataset):
    def __init__(self, data_root, transform):
        super(JsonDatasetTrain, self).__init__()
        self.data_root = data_root
        self.dir_list = listdir(data_root)
        self.transform = transform

    def __getitem__(self, index):
        # make empty data
        data_a = np.zeros((len(room_list_input)+1, 600, 600)) # silhouette
        data_b = np.zeros((len(room_list_target)+1, 600, 600)) # background

        dir = join(self.data_root, self.dir_list[index])
        for room, value in room_list_input.items():
            room_dir = join(dir, room)
            seg_name_list = listdir(room_dir)
            rand_seg_name_list = random_ins(seg_name_list)

            # if there is room
            if len(rand_seg_name_list) != 0:
                rand_seg_np = unify_ins(rand_seg_name_list, room_dir)
                data_a[value] = rand_seg_np
        
        # add silhouette to data_a[0]
        sil_path = join(dir, 'silhouette_image.png')
        sil_np = np.array(Image.open(sil_path))/255
        data_a[0] = sil_np
        
        for room, value in room_list_target.items():
            room_dir = join(dir, room)
            seg_name_list = listdir(room_dir)

            # if there is room
            if len(seg_name_list) != 0:
                seg_np = unify_ins(seg_name_list, room_dir)
                data_b[value] = seg_np

        # add background to data_b[0]
        bgr_np = sil_np.copy()
        bgr_np = np.where(bgr_np == 0, 1, 0)
        data_b[0] = bgr_np
            
        data_a = torch.tensor(data_a, dtype=torch.float32)
        data_b = torch.tensor(data_b, dtype=torch.float32)
        data_a, data_b = self.transform(data_a, data_b)

        return data_a, data_b

    def __len__(self):
        return len(self.dir_list)

class JsonDatasetTest(data.Dataset):
    def __init__(self, data_root):
        super(JsonDatasetTest, self).__init__()
        self.data_root = data_root
        self.dir_list = listdir(data_root)
        self.transform = transform

    def __getitem__(self, index):
        # make empty data
        data_a = np.zeros((len(room_list_input)+1, 600, 600)) # silhouette
        data_b = np.zeros((len(room_list_target)+1, 600, 600)) # background

        dir = join(self.data_root, self.dir_list[index])
        for room, value in room_list_input.items():
            room_dir = join(dir, room)
            seg_name_list = listdir(room_dir)
            rand_seg_name_list = random_ins(seg_name_list)

            # if there is room
            if len(rand_seg_name_list) != 0:
                rand_seg_np = unify_ins(rand_seg_name_list, room_dir)
                data_a[value] = rand_seg_np
        
        # add silhouette to data_a[0]
        sil_path = join(dir, 'silhouette_image.png')
        sil_np = np.array(Image.open(sil_path))/255
        data_a[0] = sil_np
        
        for room, value in room_list_target.items():
            room_dir = join(dir, room)
            seg_name_list = listdir(room_dir)

            # if there is room
            if len(seg_name_list) != 0:
                seg_np = unify_ins(seg_name_list, room_dir)
                data_b[value] = seg_np

        # add background to data_b[0]
        bgr_np = sil_np.copy()
        bgr_np = np.where(bgr_np == 0, 1, 0)
        data_b[0] = bgr_np
            
        data_a = torch.tensor(data_a, dtype=torch.float32)
        data_b = torch.tensor(data_b, dtype=torch.float32)

        return data_a, data_b, self.dir_list[index]

    def __len__(self):
        return len(self.dir_list)

def random_ins(seg_name_list):
    if len(seg_name_list) == 1:
        max_num = random.randrange(0, 2)
        seg_list = random.sample(seg_name_list, max_num)
        return seg_list
    elif len(seg_name_list) == 0:
        return seg_name_list

    seg_num = len(seg_name_list)
    max_num = random.randrange(0, int(seg_num/2)+1)
    seg_list = random.sample(seg_name_list, max_num)

    return seg_list

def unify_ins(seg_name_list, room_dir):
    empty_np = np.zeros((600, 600))
    for seg_name in seg_name_list:
        seg_path = join(room_dir, seg_name)
        seg_img = Image.open(seg_path)
        seg_np = np.array(seg_img)/255
        if seg_np.shape[0] != 600:
            os.remove(seg_path)
            continue
        empty_np += seg_np
    return empty_np


from torchvision.transforms.functional import rotate
from torchvision.transforms.functional import hflip, vflip
import random
def transform(input, target):
    rotate_list = [0,1,2,3]
    flip_list = [0,1,2]

    rotate_num = random.choice(rotate_list)
    flip_num = random.choice(flip_list)

    if rotate_num == 0:
        rotate_input = input
        rotate_target = target
    elif rotate_num == 1:
        rotate_input = rotate(input, angle=90)
        rotate_target = rotate(target, angle=90)
    elif rotate_num == 2:
        rotate_input = rotate(input, angle=180)
        rotate_target = rotate(target, angle=180)
    elif rotate_num == 3:
        rotate_input = rotate(input, angle=270)
        rotate_target = rotate(target, angle=270)
    
    if flip_num == 0:
        flip_input = rotate_input
        flip_target = rotate_target
    elif flip_num == 1:
        flip_input = hflip(rotate_input)
        flip_target = hflip(rotate_target)
    elif flip_num == 2:
        flip_input = vflip(rotate_input)
        flip_target = vflip(rotate_target)
    
    return flip_input, flip_target
    
