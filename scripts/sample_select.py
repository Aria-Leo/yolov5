import torch
import numpy as np
import pandas as pd
import os
import cv2
import glob
import shutil
import re
from tqdm import tqdm
from torch.nn.functional import adaptive_avg_pool2d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scripts.inception import InceptionV3


def get_feature_vector(data, cuda=False, dims=2048, batch_size=10):
    """

    Args:
        data: (batch_size, height, width, channels)(RGB)
        cuda:
        dims:
        batch_size:

    Returns:

    """
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    if batch_size is None:
        batch_size = data.shape[0]
    pred_arr = np.empty((data.shape[0], dims))
    for i in tqdm(range(0, data.shape[0], batch_size)):
        start = i
        end = i + batch_size

        images = data[start: end].astype(np.float32)

        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    return pred_arr


def select(data, image_path_arr, out_path, select_counts=None):
    features = get_feature_vector(data, cuda=True, batch_size=20)
    print('features: \n', features)
    choose_clusters, choose_ss = select_counts, 0  # 通过轮廓系数选定最优的聚类数
    if select_counts is None:
        for c in range(int(features.shape[0] / 2), features.shape[0]):
            agg = KMeans(n_clusters=c)
            pre = agg.fit_predict(features)
            ss = silhouette_score(features, pre)
            if ss > choose_ss:
                choose_ss = ss
                choose_clusters = c
    print(f'choose clusters: {choose_clusters}')
    agg = KMeans(n_clusters=choose_clusters)
    pre = agg.fit_predict(features)

    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path)
    for c in range(choose_clusters):
        class_idx = np.argwhere(pre == c).flatten()
        select_image_path = np.random.permutation(image_path_arr[class_idx])[0]
        # select_image_name = re.split(r'[\\/]', select_image_path)[-1]
        shutil.move(select_image_path, out_path)


def main():
    test_path = r'D:\demo\yolov5\runs\detect\pressure_plate_all\cluster_6'
    out_path = r'D:\demo\yolov5\runs\detect\pressure_plate_all\cluster_6_select'
    # image_path1 = os.path.join(test_path, 'imageName_1656313210_53_21_1660091505464.jpg')
    # image_path2 = os.path.join(test_path, 'imageName_1656313210_77_21_1660200097300.jpg')
    # img1 = cv2.imread(image_path1)[:, :, ::-1]
    # img2 = cv2.imread(image_path2)[:, :, ::-1]
    image_path_arr = np.array(glob.glob(os.path.join(test_path, '*')))
    data = [cv2.imread(i)[:, :, ::-1] for i in image_path_arr]
    min_size = np.min([np.min(im.shape[:1]) for im in data])
    data = np.array([cv2.resize(im, (min_size, min_size)) for im in data])
    print('data shape: ', data.shape)
    select(data, image_path_arr, out_path, select_counts=50)


if __name__ == '__main__':
    main()
