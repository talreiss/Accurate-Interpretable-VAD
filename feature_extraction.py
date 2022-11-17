import torch
import torchvision
import numpy as np
import argparse
import cv2
import os
from tqdm import tqdm
from scipy.ndimage import uniform_filter
from video_dataset import VideoDatasetWithFlows
from models import CLIP

def extract_velocity(flow, magnitude, orientation, orientations=8, motion_threshold=0.):
    orientation *= (180 / np.pi)

    cy, cx = flow.shape[:2]

    orientation_histogram = np.zeros(orientations)
    subsample = np.index_exp[cy // 2:cy:cy, cx // 2:cx:cx]
    for i in range(orientations):

        temp_ori = np.where(orientation < 360 / orientations * (i + 1),
                            orientation, -1)

        temp_ori = np.where(orientation >= 360 / orientations * i,
                            temp_ori, -1)

        cond2 = (temp_ori > -1) * (magnitude >= motion_threshold)
        temp_mag = np.where(cond2, magnitude, 0)
        temp_filt = uniform_filter(temp_mag, size=(cy, cx))

        orientation_histogram[i] = temp_filt[subsample]

    return orientation_histogram

def extract(args, root):
    all_bboxes_train = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_train.npy' % args.dataset_name),
                               allow_pickle=True)
    all_bboxes_test = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_test.npy' % args.dataset_name),
                              allow_pickle=True)

    if args.dataset_name == 'shanghaitech': # ShanghaiTech normalization
        all_bboxes_train_classes = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_train_classes.npy' % args.dataset_name),
                                   allow_pickle=True)
        all_bboxes_test_classes = np.load(os.path.join(root, args.dataset_name, '%s_bboxes_test_classes.npy' % args.dataset_name),
                                  allow_pickle=True)

    train_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                          train=True, sequence_length=0, all_bboxes=all_bboxes_train, normalize=True)
    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                         train=False, sequence_length=0, all_bboxes=all_bboxes_test, normalize=True)

    if args.dataset_name == 'ped2':
        bins = 1
    else:
        bins = 8

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIP(device)
    model.eval()

    train_velocity = []
    test_velocity = []
    train_feature_space = []
    test_feature_space = []

    with torch.no_grad():
        for idx in tqdm(range(len(train_dataset)), total=len(train_dataset)):
            batch, batch_flows, _ = train_dataset.__getitem__(idx)
            batch = batch[:, 0].to(device)
            batch_flows = batch_flows[:, 0].numpy()
            train_sample_velocities = []

            # START: Deep Feature Extraction
            features = model(batch)
            train_feature_space.append(features.contiguous().detach().cpu().numpy())
            # END

            frame_bbox = train_dataset.all_bboxes[idx]

            if len(frame_bbox) > 0 and args.dataset_name == 'shanghaitech':
                frame_classes = all_bboxes_train_classes[idx]
                length_y = ((frame_bbox[:,3] - frame_bbox[:,1]))
                non_person_indices = np.where(frame_classes != 0)[0]
                length_y[non_person_indices] = 1
            else:
                length_y = np.ones(1)

            for i in range(batch_flows.shape[0]):
                img_flow = np.transpose(batch_flows[i], [1, 2, 0])
                # convert from cartesian to polar
                _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
                mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)   # L1 Magnitudes
                mag = mag / length_y[i] if args.dataset_name == 'shanghaitech' else mag    # ShanghaiTech normalization
                velocity_cur = extract_velocity(img_flow, mag, ang, orientations=bins)
                train_sample_velocities.append(velocity_cur[None])

            train_sample_velocities = np.concatenate(train_sample_velocities, axis=0)

            train_velocity.append(train_sample_velocities)
        train_velocity = np.array(train_velocity)

        np.save('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), train_velocity)
        np.save('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), train_feature_space)

        for idx in tqdm(range(len(test_dataset)), total=len(test_dataset)):
            batch, batch_flows, _ = test_dataset.__getitem__(idx)
            batch = batch[:, 0].to(device)
            batch_flows = batch_flows[:, 0].numpy()

            # START: Deep Feature Extraction
            features = model(batch)
            test_feature_space.append(features.contiguous().detach().cpu().numpy())
            # END
            test_sample_velocities = []

            frame_bbox = test_dataset.all_bboxes[idx]

            if len(frame_bbox) > 0 and args.dataset_name == 'shanghaitech':
                frame_classes = all_bboxes_test_classes[idx]
                length_y = ((frame_bbox[:,3] - frame_bbox[:,1]))
                non_person_indices = np.where(frame_classes != 0)[0]
                length_y[non_person_indices] = 1
            else:
                length_y = np.ones(1)

            for i in range(batch_flows.shape[0]):
                img_flow = np.transpose(batch_flows[i], [1, 2, 0])
                # convert from cartesian to polar
                _, ang = cv2.cartToPolar(img_flow[..., 0], img_flow[..., 1])
                mag = np.sqrt(img_flow[..., 0] ** 2) + np.sqrt(img_flow[..., 1] ** 2)
                mag = mag / length_y[i] if args.dataset_name == 'shanghaitech' else mag   # ShanghaiTech normalization
                velocity_cur = extract_velocity(img_flow, mag, ang, orientations=bins)
                test_sample_velocities.append(velocity_cur[None])

            test_sample_velocities = np.concatenate(test_sample_velocities, axis=0)

            test_velocity.append(test_sample_velocities)

    test_velocity = np.array(test_velocity)

    np.save('extracted_features/{}/test/velocity.npy'.format(args.dataset_name), test_velocity)
    np.save('extracted_features/{}/test/deep_features.npy'.format(args.dataset_name), test_feature_space)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    args = parser.parse_args()
    root = 'data/'
    extract(args, root)