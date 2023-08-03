# encoding: utf-8
"""
#@file: number_recognition.py
#@time: 2022-07-12 18:58
#@author: ywtai
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""

import base64
import cv2
import os
import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
from utils.metrics_extra import nms


class PressurePointerRecognition:

    def __init__(self, model_path=None, model_name=None, conf=0.5, iou=0.25):
        self.model_path = model_path
        self.model_name = model_name
        if model_path is not None and model_name is not None:
            self.model = torch.hub.load(model_path, 'custom',
                                        path=os.path.join(model_path, 'models_pt', model_name),
                                        source='local')
        else:
            self.model = None
        self.model.conf = conf
        self.model.iou = iou

    def predict(self, image, inference_size=640):
        """
        根据输入的图片返回指针识别读数
        输入图片为表盘识别的输出结果
        Args:
            image: RGB format, shape=(height, width, 3)
            inference_size: 推理时输入的图片大小

        Returns:
            predict_df, scale_numbers, res
            predict_df: pd.DataFrame，包含每个数字识别框的xywh和置信度等信息
            scale_numbers: list 表盘内的刻度值序列（顺时针顺序存储）
            res: float 识别结果

        """
        result = self.model(image, size=inference_size, augment=True)
        predict_df, scale_numbers, res, base64_str = result.pandas().xywhn[0], None, None, None
        predict_df = nms(predict_df)
        # 检查表盘中心、半指针、刻度和数字是否都存在，若不存在则不予检测
        valid_flag = 1
        if predict_df.empty:
            valid_flag = 0
        need_type = ['center', 'semi-pointer', 'scale']
        for t in need_type:
            if valid_flag == 1 and t not in predict_df['name'].values:
                valid_flag = 0
                break
        # 检测是否存在数字
        numbers = predict_df[~predict_df['name'].isin(need_type)][['xcenter', 'ycenter', 'name']]
        if numbers.empty:
            valid_flag = 0
        if valid_flag == 1:
            # 寻找表盘中心点和指针尖点坐标
            center_c = predict_df[predict_df['name'] == 'center'][['xcenter', 'ycenter']].values[0]
            pointer_xywh = predict_df[predict_df['name'] == 'semi-pointer'][
                ['xcenter', 'ycenter', 'width', 'height']].values[0]
            if pointer_xywh[0] <= center_c[0] and pointer_xywh[1] <= center_c[1]:
                pointer_tip_c = np.array([pointer_xywh[0] - 1 / 2 * pointer_xywh[2],
                                          pointer_xywh[1] - 1 / 2 * pointer_xywh[3]])
            elif pointer_xywh[0] <= center_c[0] and pointer_xywh[1] > center_c[1]:
                pointer_tip_c = np.array([pointer_xywh[0] - 1 / 2 * pointer_xywh[2],
                                          pointer_xywh[1] + 1 / 2 * pointer_xywh[3]])
            elif pointer_xywh[0] > center_c[0] and pointer_xywh[1] <= center_c[1]:
                pointer_tip_c = np.array([pointer_xywh[0] + 1 / 2 * pointer_xywh[2],
                                          pointer_xywh[1] - 1 / 2 * pointer_xywh[3]])
            else:
                pointer_tip_c = np.array([pointer_xywh[0] + 1 / 2 * pointer_xywh[2],
                                          pointer_xywh[1] + 1 / 2 * pointer_xywh[3]])

            # 获取数字和中心坐标点
            # 使用k-means数字聚类
            features = numbers[['xcenter', 'ycenter']].values
            labels = numbers['name'].values
            choose_clusters, choose_ss = 1, 0  # 通过轮廓系数选定最优的聚类数
            for c in range(2, min(labels.shape[0], 16)):
                agg = KMeans(n_clusters=c)
                pre = agg.fit_predict(features)
                ss = silhouette_score(features, pre)
                if ss > choose_ss:
                    choose_ss = ss
                    choose_clusters = c
            print(f'choose clusters: {choose_clusters}')
            agg = KMeans(n_clusters=choose_clusters)
            pre = agg.fit_predict(features)
            # 相近的数字和刻度进行合并
            scale_numbers, scale_c = [], []
            scales = predict_df[predict_df['name'] == 'scale'][['xcenter', 'ycenter']].values
            for i in range(choose_clusters):
                ln_idx = np.argwhere(pre == i).flatten()
                tf, tl = features[ln_idx], labels[ln_idx]
                sort_idx = np.argsort(tf[:, 0].flatten())
                res_arr = tl[sort_idx]
                if '.' not in res_arr and res_arr[0] == '0' and len(res_arr) > 1:
                    res_arr = res_arr[0] + '.' + res_arr[1:]
                scale_numbers.append(float(''.join(res_arr)))
                # 寻找离该数字最近的刻度
                avg_sc = tf.mean(axis=0)
                choose_idx = np.argmin(np.sum((scales - avg_sc) ** 2, axis=1))
                scale_c.append(scales[choose_idx])

            scale_numbers, scale_c = np.array(scale_numbers), np.array(scale_c)
            # 根据数据规律，对识别数字进行修正(假设能够识别到刻度和数字，数字内容可以不准确)
            scale_angle = np.array(
                [np.arccos((sc[1] - center_c[1]) / np.sqrt(np.sum((sc - center_c) ** 2))) for sc in scale_c])
            scale_angle = np.array([scale_angle[i] if scale_c[i][0] <= center_c[0]
                                    else 2 * np.pi - scale_angle[i] for i in
                                    range(len(scale_angle))])
            # 夹角从小到大排序，相当于按照顺时针给刻度值排序
            angle_sort_idx = np.argsort(scale_angle)
            scale_numbers, scale_c = scale_numbers[angle_sort_idx], scale_c[angle_sort_idx]
            # 修正，计算差值
            diff_sn = np.diff(scale_numbers)
            diff_str = np.array([f'{x:.2f}' for x in diff_sn])
            count_diff_str = Counter(diff_str)
            print(scale_numbers)
            if len(count_diff_str) > 1:  # 说明存在刻度值识别错误
                print('刻度值存在错误，需要校正')
                interval = count_diff_str.most_common()[0][0]
                inter_val_first_pos = np.argwhere(diff_str == interval).flatten()[0]
                start_n = scale_numbers[inter_val_first_pos] - inter_val_first_pos * float(interval)
                end_n = scale_numbers[inter_val_first_pos] + (choose_clusters - 1 - inter_val_first_pos) * float(interval)
                scale_numbers = np.linspace(start_n, end_n, choose_clusters)
                print(scale_numbers)

            # 寻找距离指针尖点最近的刻度, 以及刻度最近的刻度数字
            choose_idx = np.argsort(np.sum((scale_c - pointer_tip_c) ** 2, axis=1))[:2]
            use_scale_c = scale_c[choose_idx]
            use_scale_numbers = scale_numbers[choose_idx]
            sort_idx = np.argsort(use_scale_numbers)
            use_scale_numbers = use_scale_numbers[sort_idx]
            use_scale_c = use_scale_c[sort_idx]

            # 计算中心点到刻度、中心点到针尖的直线斜率
            slope_tip = (pointer_tip_c[1] - center_c[1]) / (pointer_tip_c[0] - center_c[0])
            slope_scale_left = (use_scale_c[0][1] - center_c[1]) / (use_scale_c[0][0] - center_c[0])
            slope_scale_right = (use_scale_c[1][1] - center_c[1]) / (use_scale_c[1][0] - center_c[0])
            # 计算指针左边刻度到指针的夹角
            angle_left_tip = np.arccos(
                (1 + slope_scale_left * slope_tip) / np.sqrt(1 + slope_scale_left ** 2) / np.sqrt(1 + slope_tip ** 2))
            angle_left_right = np.arccos(
                (1 + slope_scale_left * slope_scale_right) / np.sqrt(1 + slope_scale_left ** 2) / np.sqrt(
                    1 + slope_scale_right ** 2))

            res = (use_scale_numbers[0] + angle_left_tip / angle_left_right
                   * (use_scale_numbers[1] - use_scale_numbers[0]))

        return predict_df, scale_numbers, res

    @staticmethod
    def check_result(scale_numbers, res):
        """
        检查刻度和读数识别是否准确
        Args:
            scale_numbers:
            res:

        Returns:

        """
        abnormal_details, status_code = '', 0
        if res is None:
            abnormal_details = '无识别结果，表盘中心、指针或刻度值缺失'
            status_code = 1
        else:
            if len(scale_numbers) <= 3:
                abnormal_details = '识别刻度值个数小于等于3个，刻度缺失'
                status_code = 2
        return abnormal_details, status_code


class PressurePlateRecognition:

    def __init__(self, model_path=None, model_name=None, conf=0.7, max_det=2):
        self.model_path = model_path
        self.model_name = model_name
        if model_path is not None and model_name is not None:
            self.model = torch.hub.load(model_path, 'custom',
                                        path=os.path.join(model_path, 'models_pt', model_name),
                                        source='local')
        else:
            self.model = None
        self.model.conf = conf
        self.model.max_det = max_det

    def predict(self, image, crop_path=None, area_coordinate=None, inference_size=640):
        """
        根据原始输入图片，返回表盘识别区域
        Args:
            image: (height, width, 3)
            crop_path: 截取表盘图片存储路径
            area_coordinate:
            inference_size:

        Returns:

        """
        result = self.model(image, size=inference_size)
        if crop_path is not None:
            crop = result.crop(save=True, save_dir=crop_path)
        else:
            crop = result.crop(save=False)
        order_crop = [s_c['im'] for s_c in crop]
        predict_df = result.pandas().xywhn[0]
        if area_coordinate is not None:
            order_crop = [None] * len(area_coordinate)
            order_coordinate = [None] * len(area_coordinate)
            for i, p_a in predict_df.iterrows():
                dist, sub_idx = np.inf, 0
                pre_xy = p_a[['xcenter', 'ycenter']].values
                for idx, sub_area in enumerate(area_coordinate):
                    sub_x, sub_y = sub_area[0], sub_area[1]
                    pre_dist = np.sqrt((pre_xy[0] - sub_x) ** 2 + (pre_xy[1] - sub_y) ** 2)
                    if pre_dist < dist:
                        dist = pre_dist
                        sub_idx = idx
                order_crop[sub_idx] = crop[i]['im']
                order_coordinate[sub_idx] = p_a[['xcenter', 'ycenter', 'width', 'height']].tolist()
        else:
            order_coordinate = [s[['xcenter', 'ycenter', 'width', 'height']].tolist()
                                for _, s in predict_df.iterrows()]

        return order_crop, order_coordinate


class PointerRecognition:

    def __init__(self, model_path, plate_model_name=None, number_model_name=None):
        self.model_path = model_path
        self.plate_model_name = plate_model_name
        self.number_model_name = number_model_name
        self.plate_model = PressurePlateRecognition(model_path, plate_model_name)
        self.number_model = PressurePointerRecognition(model_path, number_model_name)

    def predict(self, image, area_coordinate=None):
        """
        根据输入图片，返回指针读数识别结果
        表盘识别+指针读数识别封装
        Args:
            image:
            area_coordinate:

        Returns:

        """
        plate_res_list, plate_coordinate = self.plate_model.predict(
            image, area_coordinate=area_coordinate)
        status_code, number_res, predict_dfs = -1, [], []
        for plate_res in plate_res_list:
            if plate_res is None:
                number_res.append(None)
                predict_dfs.append(None)
                continue
            predict_df, scale_numbers, num = self.number_model.predict(plate_res)
            number_abnormal_info = self.number_model.check_result(scale_numbers, num)
            sub_status_code = number_abnormal_info[1]
            if status_code == -1:
                status_code = sub_status_code
            else:
                status_code = min(status_code, sub_status_code)
            number_res.append(num)
            predict_dfs.append(predict_df)

        # 图片读数标注
        cv_image = image[..., ::-1].copy()
        for xywh, num in zip(plate_coordinate, number_res):
            if xywh is None or num is None:
                continue
            x_range = (int((xywh[0] - xywh[2] / 2) * image.shape[1]),
                       int((xywh[0] + xywh[2] / 2) * image.shape[1]))
            y_range = (int((xywh[1] - xywh[3] / 2) * image.shape[0]),
                       int((xywh[1] + xywh[3] / 2) * image.shape[0]))
            cv_image = cv2.rectangle(cv_image, (x_range[0], y_range[0]),
                                     (x_range[1], y_range[1]), (0, 0, 255), 2)
            text_x = int(xywh[0] * image.shape[1])
            text_y = int((xywh[1] - xywh[3] / 2) * image.shape[0])
            shift_x = int(image.shape[1] * 0.01)
            shift_y = int(image.shape[0] * 0.01)
            cv2.putText(cv_image, f'{num:.2f}', (text_x - shift_x, text_y - shift_y),
                        cv2.FONT_HERSHEY_PLAIN, 5.0, (255, 255, 255), 5)

        base64_img = cv2.imencode('.jpg', cv_image)[1].tostring()
        base64_img = base64.b64encode(base64_img)
        base64_str = str(base64_img, 'utf-8')

        res = {
            'plate_res': plate_res_list,
            'predict_df': predict_dfs,
            'status_code': status_code,
            'number_res': number_res,
            'labeled_image': base64_str
        }
        return res


if __name__ == '__main__':
    m_path = 'D:\\demo\\recognition'
    plate_m_name = 'pressure_plate-v2.pt'
    pointer_m_name = 'pressure_pointer-v1.pt'
    abnormal_image_path = 'D:\\demo\\PressureMeterData\\noise\\test3.jpg'
    normal_image_path = 'D:\\demo\\PressureMeterData\\PlateDataV2\\images\\val\\img31.jpg'
    im = cv2.imread(normal_image_path)
    b = cv2.imencode('.jpg', im)[1]
    # b64 = str(base64.b64encode(b))[2:-1]
    b64 = str(base64.b64encode(b), 'utf-8')
    dim = base64.b64decode(b64)
    img_arr = np.frombuffer(dim, np.uint8)
    im = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)[:, :, ::-1]
    nr = PointerRecognition(m_path, plate_m_name, pointer_m_name)
    pre_res = nr.predict(im)
    for item in pre_res:
        print(item)
