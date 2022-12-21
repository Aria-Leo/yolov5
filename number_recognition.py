# encoding: utf-8
"""
#@file: number_recognition.py
#@time: 2022-07-12 18:58
#@author: ywtai
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""

import cv2
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.metrics_extra import nms


class GasNumberRecognition:

    def __init__(self, model_path=None, model_name=None, nms_conf=0.3, iou=0.25):
        self.model_path = model_path
        self.model_name = model_name
        print('model path:', os.path.join(model_path, 'models_pt', model_name))
        if model_path is not None and model_name is not None:
            self.model = torch.hub.load(model_path, 'custom',
                                        path=os.path.join(model_path, 'models_pt', model_name),
                                        source='local')
        else:
            self.model = None
        self.model.conf = nms_conf
        self.model.iou = iou

    def predict(self, image, inference_size=288):
        result = self.model(image, size=inference_size, augment=True)
        predict_df = result.pandas().xyxy[0]
        res, sort_key = [], []
        if not predict_df.empty:
            # 根据坐标构建特征对图片进行聚类，将数字分组
            features = np.array([[(s['xmax'] + s['xmin']) / 2, s['ymax'] * 5]
                                 for _, s in predict_df.iterrows()])
            labels = predict_df['name'].values
            # choose_clusters = 6
            choose_clusters, choose_ss = 1, 0  # 通过轮廓系数选定最优的聚类数
            for c in range(2, min(labels.shape[0] - 1, 11)):
                agg = KMeans(n_clusters=c)
                pre = agg.fit_predict(features)
                ss = silhouette_score(features, pre)
                if ss > choose_ss:
                    choose_ss = ss
                    choose_clusters = c
            print(f'choose clusters: {choose_clusters}')
            agg = KMeans(n_clusters=choose_clusters)
            pre = agg.fit_predict(features)
            # 存储识别结果
            for i in range(choose_clusters):
                ln_idx = np.argwhere(pre == i).flatten()
                ln_sort_idx = np.argsort(features[ln_idx, 0].flatten())
                res_arr = labels[ln_idx][ln_sort_idx]
                res_str = ''.join(res_arr)
                res.append(res_str)
                temp_key = features[ln_idx][ln_sort_idx][-1]
                sort_key.append((temp_key[0], temp_key[1]))
            sort_key = np.array(sort_key, dtype=[('x', '<f8'), ('y', '<f8')])
            sort_idx = np.argsort(sort_key, order=('y', 'x'))
            res = np.array(res, dtype='U25')[sort_idx]
        else:
            res = np.array(res)
        print(predict_df)
        print(res)
        return predict_df, res

    def predict_opt(self, image, inference_size=288):
        result = self.model(image, size=inference_size, augment=True)
        predict_df = result.pandas().xywhn[0]
        predict_df = nms(predict_df)
        pre_df, res = pd.DataFrame(), np.array([])
        if not predict_df.empty:
            # 根据坐标构建特征对图片进行聚类，将数字分组
            features = np.array([[s['xcenter'], (s['ycenter'] + s['height'] / 2) * 5]
                                 for _, s in predict_df.iterrows()])
            labels = predict_df['name'].values
            # choose_clusters = 6
            choose_clusters, choose_ss = 1, 0  # 通过轮廓系数选定最优的聚类数
            for c in range(2, min(labels.shape[0] - 1, 11)):
                agg = KMeans(n_clusters=c)
                pre = agg.fit_predict(features)
                ss = silhouette_score(features, pre)
                if ss > choose_ss:
                    choose_ss = ss
                    choose_clusters = c
            print(f'choose clusters: {choose_clusters}')
            agg = KMeans(n_clusters=choose_clusters)
            pre = agg.fit_predict(features)
            # 存储识别结果
            for i in range(choose_clusters):
                ln_idx = np.argwhere(pre == i).flatten()
                predict_df.loc[ln_idx, 'cluster_id'] = i

            # 对dataframe进行排序
            pre_group = predict_df.groupby(['cluster_id'])
            pre_df, cluster_sort_key = pd.DataFrame(), []
            for c, df in pre_group:
                cluster_sort_key.append((df['xcenter'].min(), df['ycenter'].min()))
            cluster_sort_key = np.array(cluster_sort_key, dtype=[('x', '<f8'), ('y', '<f8')])
            sort_idx = np.argsort(cluster_sort_key, order=('y', 'x'))
            res = []
            for i in range(choose_clusters):
                temp = predict_df[predict_df['cluster_id'] == sort_idx[i]]
                temp = temp.sort_values(by=['xcenter'])
                temp['cluster_id'] = i
                pre_df = pd.concat([pre_df, temp], ignore_index=True, sort=False)
                res.append(''.join(temp['name'].tolist()))
            res = np.array(res, dtype='U25')

        print(pre_df)
        print(res)

        return pre_df, res

    @staticmethod
    def check_result(predict_df, res, num_conf_threshold=0.8, max_number=6,
                     number_length_min=(10, 3, 3, 2, 4, 6),
                     float_places=(4, 2, 2, 1, 1, 2)):
        # 主要校验小数位和数字长度的合法性
        abnormal_details, status_code = '', 0
        if len(res) == 0:
            abnormal_details = '无识别结果'
            status_code = -1
        else:
            # 能够全部识别数字时才对小数位进行校正，否则只对总量进行校正
            if len(res) == max_number:
                iter_counts = len(res)
            else:
                iter_counts = 1
            for i in range(iter_counts):
                current = res[i]
                if '.' not in current:
                    if len(current) < number_length_min[i] and status_code < 7:
                        abnormal_details = '存在数字识别不准确，小数点未识别的情况'
                        status_code = 7
                    else:
                        # 对小数点进行补偿
                        temp = current[:-float_places[i]] + '.' + current[-float_places[i]:]
                        res[i] = temp
                        if status_code < 3:
                            abnormal_details = '存在数字识别准确，小数点未识别的情况'
                            status_code = 3
                else:
                    # 检验小数点个数是否合法
                    point_counts = current.count('.')
                    if point_counts > 1 or (
                            point_counts == 1 and current.index('.') != (len(current) - float_places[i] - 1)):
                        # 对小数点进行修正
                        temp = current.replace('.', '')
                        temp = temp[:-float_places[i]] + '.' + temp[-float_places[i]:]
                        res[i] = temp
                        if status_code < 3:
                            abnormal_details = '存在数字识别准确，小数点识别不准确的情况'
                            status_code = 3

                    elif i == 0 and len(current) - 1 < number_length_min[i] and status_code < 6:
                        abnormal_details = '存在数字识别不准确，小数点已识别的情况'
                        status_code = 6

                # 针对第一个数进行更进一步的校正，当识别数字个数超过给定阈值时，剔除置信度最低的那部分数字
                # 若剔除后其他数字置信度依旧低于阈值，则舍弃该组结果
                current = res[i]
                current_df = predict_df[predict_df['cluster_id'] == i]
                current_df.index = np.arange(len(current_df))
                if len(current) - 1 > number_length_min[i] and status_code < 4:
                    if i == 0:
                        sort_idx = current_df[current_df['name'] != '.'].sort_values(by=['confidence']).index.values
                        need_drop = len(current) - 1 - number_length_min[i]
                        drop_idx, cs = sort_idx[:need_drop], ''
                        for idx in range(len(current)):
                            if idx in drop_idx or current[idx] == '.':
                                continue
                            cs += current[idx]
                        cs = cs[:-float_places[i]] + '.' + cs[-float_places[i]:]
                        res[i] = cs
                        abnormal_details = '总量已校正，识别数字多于期望个数'
                        status_code = 4

                    else:  # 除了第一行的数字外，如果存在数字大于指定个数且某个数字置信度比其他数字差别较大的也去掉
                        use_idx = current_df[current_df['name'] != '.'].sort_values(by=['confidence']).index.values
                        need_check = len(current) - 1 - number_length_min[i]
                        normal_idx = use_idx[need_check:]
                        cfp = current_df.iloc[normal_idx]['confidence'].min() - 0.1
                        drop_idx = current_df[(current_df['name'] != '.') & (current_df['confidence'] < cfp)].index.values
                        if len(drop_idx) > 0:
                            cs = ''
                            for idx in range(len(current)):
                                if idx in drop_idx or current[idx] == '.':
                                    continue
                                cs += current[idx]
                            cs = cs[:-float_places[i]] + '.' + cs[-float_places[i]:]
                            res[i] = cs
                            abnormal_details = '总量已校正，存在其他数字多于期望个数'
                            status_code = 4
                    # 删除对应的number_df中置信度低的数字
                    if len(drop_idx) > 0:
                        current_df = current_df.drop(index=drop_idx)

                # 针对第一行的结果，必须保证所有的数字置信度大于指定阈值且个数符合条件才输出
                if i == 0 and (not current_df[(current_df['name'] != '.')
                                              & (current_df['confidence'] < num_conf_threshold)].empty
                               or len(res[i]) - 1 != number_length_min[i]):
                    abnormal_details = '识别数字不等于期望个数，或校正后的数字置信度依旧低于指定阈值'
                    status_code = 5

            if iter_counts == 1 and status_code == 0:
                abnormal_details = '总量已校正，但存在整组数字未识别或识别组数超过最大值的情况'
                status_code = 2

            # 检验置信度
            if not predict_df[predict_df['confidence'] < num_conf_threshold - 0.2].empty and status_code == 0:
                abnormal_details = '存在置信度低于阈值的识别数字'
                status_code = 1

            # 当检验数字个数超过max_number时，只保留置信度最高的几组
            # if len(res) > max_number:
            #     confidence_s = predict_df.groupby('cluster_id').mean()['confidence']
            #     class_sort_idx = confidence_s.sort_values(ascending=False)[:max_number].index.values
            #     res = res[class_sort_idx]
            #     predict_df = predict_df.iloc[class_sort_idx]

        return abnormal_details, status_code


class GasPlateRecognition:

    def __init__(self, model_path=None, model_name=None, nms_conf=0.7, max_det=1):
        self.model_path = model_path
        self.model_name = model_name
        if model_path is not None and model_name is not None:
            self.model = torch.hub.load(model_path, 'custom',
                                        path=os.path.join(model_path, 'models_pt', model_name),
                                        source='local')
        else:
            self.model = None
        self.model.conf = nms_conf
        self.model.max_det = max_det

    def predict(self, image, crop_path=None, inference_size=800):
        result = self.model(image, size=inference_size)
        if crop_path is not None:
            crop = result.crop(save=True, save_dir=crop_path)
        else:
            crop = result.crop(save=False)
        res = np.array([])
        if len(crop) > 0:
            res = crop[0]['im']
        return res

    @staticmethod
    def check_result(res):
        print('plate shape: ', res.shape)
        abnormal_details, status_code = '', 0
        if len(res) == 0:
            abnormal_details = '无识别结果'
            status_code = -1
        return abnormal_details, status_code


class NumberRecognition:

    def __init__(self, model_path, plate_model_name=None, number_model_name=None):
        self.model_path = model_path
        self.plate_model_name = plate_model_name
        self.number_model_name = number_model_name
        self.plate_model = GasPlateRecognition(model_path, plate_model_name)
        self.number_model = GasNumberRecognition(model_path, number_model_name)

    def predict(self, image):
        """

        Args:
            image: RGB format

        Returns:

        """
        plate_res = self.plate_model.predict(image)
        # if len(plate_res) > 0:
        #     if not isinstance(image, str):
        #         output_res = plate_res[:, :, ::-1]
        #     else:
        #         output_res = plate_res
        #     cv2.imwrite('data/images/current_gas_plate.jpg', output_res)
        plate_abnormal_info = self.plate_model.check_result(plate_res)
        status_code, number_res = 0, []
        predict_df = pd.DataFrame()
        if plate_abnormal_info[1] == 0:
            predict_df, number_res = self.number_model.predict_opt(plate_res)
            number_abnormal_info = self.number_model.check_result(predict_df, number_res)
            status_code = number_abnormal_info[1]
        else:
            status_code = plate_abnormal_info[1]
        return predict_df, status_code, list(number_res)


if __name__ == '__main__':
    m_path = 'D:\\demo\\recognition'
    plate_m_name = 'gas_plate.pt'
    number_m_name = 'gas_number-v3.pt'
    normal_image_path = 'D:\\demo\\GasMeterData_pre\\active\\abnormal\\images\\img202210172302091608.jpg'
    test_image_path = 'D:\\demo\\yolov5\\data\\images\\current_gas_plate.jpg'
    number_model = GasNumberRecognition(m_path, number_m_name)
    number_model.predict_opt(normal_image_path)
    # nr = NumberRecognition(m_path, plate_m_name, number_m_name)
    # pre_res = nr.predict(normal_image_path)
    # print(pre_res)
