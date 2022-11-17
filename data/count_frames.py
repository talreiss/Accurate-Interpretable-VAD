import os
import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
args = parser.parse_args()
root = '{}/training/frames/'.format(args.dataset_name)

train_clip_lengths = []
test_clip_lengths = []

folders = glob.glob(os.path.join(root, '*'))
folders.sort()
lengths = []
count = 0
for i, folder in enumerate(folders):
    video_folder = folder.split('/')[-1]
    video_frames = glob.glob(os.path.join(root, video_folder, '*.jpg'))
    video_frames.sort()
    count += len(video_frames)
    train_clip_lengths.append(count)

root = '{}/testing/frames/'.format(args.dataset_name)


folders = glob.glob(os.path.join(root, '*'))
folders.sort()
lengths = []
count = 0
for i, folder in enumerate(folders):
    video_folder = folder.split('/')[-1]
    video_frames = glob.glob(os.path.join(root, video_folder, '*.jpg'))
    video_frames.sort()
    count += len(video_frames)
    test_clip_lengths.append(count)

np.save(args.dataset_name + '/train_clip_lengths.npy', train_clip_lengths)
np.save(args.dataset_name + '/test_clip_lengths.npy', test_clip_lengths)