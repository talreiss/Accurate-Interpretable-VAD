import numpy as np
import argparse
import faiss
from video_dataset import VideoDatasetWithFlows, img_tensor2numpy
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from scipy.ndimage import gaussian_filter1d
from sklearn.mixture import GaussianMixture
import sys
from sklearn.mixture import GaussianMixture

def gaussian_video(video, lengths, sigma=3):
    scores = np.zeros_like(video)
    prev = 0
    for cur in lengths:
        scores[prev: cur] = gaussian_filter1d(video[prev: cur], sigma)
        prev = cur
    return scores

def macro_auc(video, test_labels, lengths):
    prev = 0
    auc = 0
    for i, cur in enumerate(lengths):
        cur_auc = roc_auc_score(np.concatenate(([0], test_labels[prev: cur], [1])),
                             np.concatenate(([0], video[prev: cur], [sys.float_info.max])))
        auc += cur_auc
        prev = cur
    return auc / len(lengths)


def evaluate(args, root):
    test_clip_lengths = np.load(os.path.join(root, args.dataset_name, 'test_clip_lengths.npy'))

    train_velocity = np.load('extracted_features/{}/train/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    train_velocity = np.concatenate(train_velocity, 0)
    train_deep_features = np.load('extracted_features/{}/train/deep_features.npy'.format(args.dataset_name), allow_pickle=True)
    train_deep_features = np.concatenate(train_deep_features, 0)

    train_pose = np.load('extracted_features/{}/train/pose.npy'.format(args.dataset_name), allow_pickle=True)
    without_empty_frames = []
    for i in tqdm(range(len(train_pose))):
        if len(train_pose[i]):
            without_empty_frames.append(train_pose[i])
    train_pose = np.concatenate(without_empty_frames, 0)

    test_velocity = np.load('extracted_features/{}/test/velocity.npy'.format(args.dataset_name), allow_pickle=True)
    test_pose = np.load('extracted_features/{}/test/pose.npy'.format(args.dataset_name), allow_pickle=True)
    test_deep_features = np.load('extracted_features/{}/test/deep_features.npy'.format(args.dataset_name), allow_pickle=True)

    test_dataset = VideoDatasetWithFlows(dataset_name=args.dataset_name, root=root,
                                         train=False, sequence_length=0, all_bboxes=None, normalize=False, mode='last')

    if args.dataset_name == 'ped2':
        velocity_density_estimator = GaussianMixture(n_components=2, random_state=0).fit(train_velocity)
    else:
        velocity_density_estimator = GaussianMixture(n_components=5, random_state=0).fit(train_velocity)

    train_velocity_scores = -velocity_density_estimator.score_samples(train_velocity)

    train_pose_scores = np.load('extracted_features/{}/train_pose_scores.npy'.format(args.dataset_name))
    train_deep_features_scores = np.load('extracted_features/{}/train_deep_features_scores.npy'.format(args.dataset_name))

    min_deep_features = np.min(train_deep_features_scores)
    max_deep_features = np.max(train_deep_features_scores)
    # max_deep_features = np.percentile(train_deep_features_scores, 99.9)

    min_pose = np.min(train_pose_scores)
    max_pose = np.percentile(train_pose_scores, 99.9)

    min_velocity = np.min(train_velocity_scores)
    max_velocity = np.percentile(train_velocity_scores, 99.9)

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(train_deep_features.shape[1])
    index_deep_features = faiss.index_cpu_to_gpu(res, 0, index)
    index_deep_features.add(train_deep_features.astype(np.float32))

    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatL2(train_pose.shape[1])
    index_pose = faiss.index_cpu_to_gpu(res, 0, index)
    index_pose.add(train_pose.astype(np.float32))

    test_velocity_scores = []
    test_deep_features_scores = []
    test_pose_scores = []

    for i in tqdm(range(len(test_dataset)), total=len(test_dataset)):
        cur_pose = test_pose[i]
        cur_velocity = test_velocity[i]
        cur_deep_features = test_deep_features[i]

        if cur_pose.shape[0]:
            D, I = index_pose.search(cur_pose.astype(np.float32), 1)
            score_pose = np.mean(D, axis=1)
            max_score_pose = np.max(score_pose)
            test_pose_scores.append(max_score_pose)
        else:
            test_pose_scores.append(0)

        D, I = index_deep_features.search(cur_deep_features.astype(np.float32), 1)
        score_features = np.mean(D, axis=1)
        max_score_features = np.max(score_features)
        test_deep_features_scores.append(max_score_features)

        max_score_velocity = np.max(-velocity_density_estimator.score_samples(cur_velocity))
        test_velocity_scores.append(max_score_velocity)

    test_velocity_scores = np.array(test_velocity_scores)
    test_deep_features_scores = np.array(test_deep_features_scores)
    test_pose_scores = np.array(test_pose_scores)

    test_velocity_scores = (test_velocity_scores - min_velocity) / (max_velocity - min_velocity)
    test_pose_scores = (test_pose_scores - min_pose) / (max_pose - min_pose)
    test_deep_features_scores = (test_deep_features_scores - min_deep_features) / (max_deep_features - min_deep_features)

    if args.dataset_name == 'shanghaitech':
        final_scores = gaussian_video(test_velocity_scores + test_pose_scores,
                                      test_clip_lengths, sigma=args.sigma)
    else:
        final_scores = gaussian_video(test_velocity_scores + test_pose_scores + test_deep_features_scores,
                                      test_clip_lengths, sigma=args.sigma)

    print('Micro AUC: ', roc_auc_score(test_dataset.all_gt, final_scores) * 100)
    print('Macro AUC: ', macro_auc(final_scores, test_dataset.all_gt, test_clip_lengths) * 100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="ped2", help='dataset name')
    parser.add_argument("--sigma", type=int, default=3, help='sigma for gaussian1d smoothing')
    args = parser.parse_args()
    root = 'data/'
    evaluate(args, root)
