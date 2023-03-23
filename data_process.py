# encoding: utf-8
"""
#@file: data_process.py
#@time: 2022-07-21 16:05
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
import os
# import yaml
import shutil
import numpy as np
import model_cfg
# import train
from apscheduler.schedulers.background import BackgroundScheduler


class ALDetector:

    @staticmethod
    def abnormal_counts(data_type='gas'):
        """
        判断图片数量是否超过阈值，超过则删除时间较早的图片
        Returns:

        """
        print('start abnormal counts detecting')
        save_folder = model_cfg.abnormal_save_folder_gas
        if data_type == 'pressure':
            save_folder = model_cfg.abnormal_save_folder_pressure
        abnormal_save_image_path = os.path.join(model_cfg.model_path, save_folder, 'images')
        abnormal_save_label_path = os.path.join(model_cfg.model_path, save_folder, 'labels')
        if os.path.exists(abnormal_save_image_path) and os.path.exists(abnormal_save_label_path):
            img_list = os.listdir(abnormal_save_image_path)
            mark_idx = np.argsort([s.replace('.jpg', '').replace('img', '') for s in img_list])
            if len(img_list) > model_cfg.abnormal_counts_limit:
                delete_counts = len(img_list) - model_cfg.abnormal_counts_limit
                delete_arr = np.array(img_list)[mark_idx][:delete_counts]
                for fi in delete_arr:
                    os.remove(os.path.join(abnormal_save_image_path, fi))
                    os.remove(os.path.join(abnormal_save_label_path, fi.replace('jpg', 'txt')))
        print('detecting over')

    def run(self):
        # job_defaults = {
        #     'coalesce': True,
        #     'misfire_grace_time': None
        # }
        scheduler = BackgroundScheduler(timezone='Asia/Shanghai')
        scheduler.add_job(self.abnormal_counts, 'cron', hour='7-19', minute='*/5')
        scheduler.add_job(self.abnormal_counts, 'cron', hour='7-19', minute='*/5', kwargs={'data_type': 'pressure'})
        # scheduler.add_job(self.trigger_training, 'cron', hour='5-22/2')
        scheduler.start()


class DataTransform:

    @staticmethod
    def cvat_to_yolo(data_path, out_path, train_split=0.8, data_limit=(5000, 1000), select_mode='sort'):
        """

        Args:
            data_path:
            out_path:
            train_split:
            data_limit:
            select_mode:

        Returns:

        """
        doc_list = os.listdir(data_path)
        images = np.array([x for x in doc_list if '.txt' not in x])
        # 数据分为训练集和验证集（用最新的数据做验证集）
        if select_mode == 'sort':
            keys = [x.split('.')[0].lstrip('img') for x in images]
            key_idx = np.argsort(keys)
        else:
            key_idx = np.random.permutation(images.shape[0])
        t_split = int(len(images) * train_split)
        train_idx, val_idx = key_idx[:t_split], key_idx[t_split + 1:]
        train_images, val_images = images[train_idx], images[val_idx]
        train_labels = [x.replace('jpg', 'txt') for x in train_images]
        val_labels = [x.replace('jpg', 'txt') for x in val_images]
        train_image_path = os.path.join(out_path, 'images/train')
        val_image_path = os.path.join(out_path, 'images/val')
        train_label_path = os.path.join(out_path, 'labels/train')
        val_label_path = os.path.join(out_path, 'labels/val')
        for obj, obj_path in zip([train_images, val_images, train_labels, val_labels],
                                 [train_image_path, val_image_path, train_label_path, val_label_path]):
            if not os.path.exists(obj_path):
                os.makedirs(obj_path)
            for ti in obj:
                shutil.copyfile(os.path.join(data_path, ti), os.path.join(obj_path, ti))

        # 检查out_path中的数据量是否超过阈值
        train_image_list = os.listdir(train_image_path)
        val_image_list = os.listdir(val_image_path)
        if len(train_image_list) > data_limit[0]:
            if select_mode == 'sort':
                mark_idx = np.argsort([s.replace('.jpg', '').replace('img', '')
                                       for s in train_image_list])
            else:
                mark_idx = np.random.permutation(len(train_image_list))
            delete_counts = len(train_image_list) - data_limit[0]
            delete_arr = np.array(train_image_list)[mark_idx][:delete_counts]
            for fi in delete_arr:
                os.remove(os.path.join(train_image_path, fi))
                os.remove(os.path.join(train_label_path, fi.replace('jpg', 'txt')))

        if len(val_image_list) > data_limit[1]:
            if select_mode == 'sort':
                mark_idx = np.argsort([s.replace('.jpg', '').replace('img', '')
                                       for s in val_image_list])
            else:
                mark_idx = np.random.permutation(len(val_image_list))
            delete_counts = len(val_image_list) - data_limit[1]
            delete_arr = np.array(val_image_list)[mark_idx][:delete_counts]
            for fi in delete_arr:
                os.remove(os.path.join(val_image_path, fi))
                os.remove(os.path.join(val_label_path, fi.replace('jpg', 'txt')))


if __name__ == '__main__':
    input_path = r'D:\demo\PressureMeterData\mp_train2\obj_train_data'
    o_path = r'D:\demo\PressureMeterData\MeterPointerV2'
    DataTransform().cvat_to_yolo(input_path, o_path, select_mode='random')
