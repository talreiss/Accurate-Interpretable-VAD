"""
The code in this file is adapted from:
https://github.com/LiUzHiAn/hf2vad/blob/master/pre_process/extract_flows.py
"""

import argparse
import torch
import numpy as np
import cv2
import os
from tqdm import tqdm
from torch.utils.data import DataLoader

from video_dataset import VideoDataset
from pre_processing.flownet_networks.flownet2_models import FlowNet2

FLOWNET_INPUT_WIDTH = {"ped2": 512 * 2, "avenue": 512 * 2, "shanghaitech": 1024}
FLOWNET_INPUT_HEIGHT = {"ped2": 384 * 2, "avenue": 384 * 2, "shanghaitech": 640}


def extracting_flows(dataset_name, root, train):
    if train:
        of_save_dir = os.path.join(root, dataset_name, "training", "flows")
    else:
        of_save_dir = os.path.join(root, dataset_name, "testing", "flows")

    dataset = VideoDataset(dataset_name=dataset_name, root=root, train=train, sequence_length=1,
                           bboxes_extractions=True)

    WIDTH, HEIGHT = FLOWNET_INPUT_WIDTH[dataset_name], FLOWNET_INPUT_HEIGHT[dataset_name]

    flownet2 = FlowNet2()
    path = 'pre_processing/checkpoints/FlowNet2_checkpoint.pth.tar'
    pretrained_dict = torch.load(path)['state_dict']
    model_dict = flownet2.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    flownet2.load_state_dict(model_dict)
    flownet2.cuda()

    dataset_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    for idx, (batch, _) in tqdm(enumerate(dataset_loader), total=len(dataset)):
        cur_img_addr = dataset.frame_addresses[idx]
        cur_img_name = cur_img_addr.split('/')[-1]

        # path to store flows
        video_of_path = os.path.join(of_save_dir, cur_img_addr.split('/')[-2])
        if os.path.exists(video_of_path) is False:
            os.makedirs(video_of_path, exist_ok=True)

        # batch [bs,#frames,3,h,w]
        cur_imgs = np.transpose(batch[0].numpy(), [0, 2, 3, 1])  # [#frames,3,h,w] -> [#frames,h,w,3]

        old_size = (cur_imgs.shape[2], cur_imgs.shape[1])  # w,h

        # resize format (w',h')
        im1 = cv2.resize(cur_imgs[0], (WIDTH, HEIGHT))  # the frame before centric
        im2 = cv2.resize(cur_imgs[1], (WIDTH, HEIGHT))  # centric frame
        # [0-255]
        ims = np.array([im1, im2]).astype(np.float32)  # [2,h',w',3]
        ims = torch.from_numpy(ims).unsqueeze(0)
        ims = ims.permute(0, 4, 1, 2, 3).contiguous().cuda()  # [bs,2,H,W,img_channel] -> [bs,img_channel,2,H,W]

        pred_flow = flownet2(ims).cpu().data
        pred_flow = pred_flow[0].numpy().transpose((1, 2, 0))  # [h',w',2]
        new_inputs = cv2.resize(pred_flow, old_size)  # [h,w,2]

        # save new raw inputs
        np.save(os.path.join(video_of_path, cur_img_name + '.npy'), new_inputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--train", action='store_true', help='train or test data')

    args = parser.parse_args()
    root = 'data/'

    with torch.no_grad():
        extracting_flows(dataset_name=args.dataset_name, root=root, train=args.train)