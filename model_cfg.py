# encoding: utf-8
"""
#@file: model_cfg.py
#@time: 2022-07-13 13:39
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
import os

# 474902489004576: 燃气表摄像头1
# 474902498119200: 燃气表摄像头2
valid_number = ['474902489004576', '474902498119200']
base64_url = 'http://192.168.2.123/video/api/Video/CaptureJPEGPictureBase64'
model_path = os.path.dirname(os.path.abspath(__file__))
gas_plate_model = 'gas_plate.pt'
gas_number_model = 'gas_number-v3.pt'
pressure_plate_model = 'pressure_plate-v2.pt'
pressure_pointer_model = 'pressure_pointer-v3.pt'
abnormal_save_folder_gas = os.path.join('active', 'abnormal', 'gas')
abnormal_save_folder_pressure = os.path.join('active', 'abnormal', 'pressure')
abnormal_counts_limit = 500
active_learning_folder_gas = os.path.join('active', 'modified', 'gas')
active_learning_folder_pressure = os.path.join('active', 'modified', 'pressure')
active_learning_counts_limit = 100
active_learning_splits = [0.8, 0.1, 0.1]  # 标注数据集划分比例
all_data_limit = [5000, 1000, 1000]  # 训练集，验证集，测试集
gas_number_training_cfg = {
    'img': 288,
    'batch-size': 20,
    'epochs': 300,
    'data': 'GasMeterAp.yaml',
    'weights': 'models_pt/gas_number.pt',
    'optimizer': 'AdamW',
    'multi-scale': True,
    'rect': True,
    'nosave': True,
    'noplots': True
}
