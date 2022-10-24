# encoding: utf-8
"""
#@file: metrics_extra.py
#@time: 2022-07-13 11:43
#@author: ywtai
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""

import torch
import numpy as np
import pandas as pd
from .metrics import bbox_iou


def nms(df, iou_threshold=0.6) -> pd.DataFrame:
    """

    Args:
        df: pandas with columns like (xcenter, ycenter, width, height, confidence, class name, cluster_id)
        iou_threshold: iou threshold

    Returns:
        filtered df

    """
    # 按照confidence从高到低排序
    ordered_df = df.sort_values(by='confidence', ascending=False)
    ordered_df.index = np.arange(len(ordered_df))
    total_length = ordered_df.shape[0]
    deleted_index = np.array([])
    xywh_c = ['xcenter', 'ycenter', 'width', 'height']
    for i in range(total_length):
        if i in deleted_index:
            continue
        tensor_i = torch.from_numpy(np.array(ordered_df.iloc[i:i+1][xywh_c].values, dtype=np.float64))
        tensor_rest = torch.from_numpy(np.array(ordered_df.iloc[i+1:][xywh_c].values, dtype=np.float64))
        iou = bbox_iou(tensor_i, tensor_rest, DIoU=True).flatten()
        deleted = torch.nonzero(iou > iou_threshold).flatten() + i + 1
        deleted = deleted.numpy()
        if deleted.shape[0]:
            deleted_index = np.concatenate([deleted_index, ordered_df.index[deleted].values])
    deleted_index = np.unique(deleted_index)
    ordered_df = ordered_df.drop(index=deleted_index)
    ordered_df.index = np.arange(len(ordered_df))
    return ordered_df
