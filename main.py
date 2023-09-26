# encoding: utf-8
"""
#@file: main.py
#@time: 2022-07-13 11:43
#@author: ywtai 
#@contact: 632910913@qq.com
#@software: PyCharm
#@desc:
"""
import base64
import cv2
from collections import defaultdict, deque
from fastapi import FastAPI, HTTPException, Body
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块
import time
from datetime import datetime
import requests
import numpy as np
import uvicorn
import os
import model_cfg
from audio_recognition import AudioRecognition
from number_recognition import NumberRecognition
from pointer_recognition import PointerRecognition
from temperature_recognition import TemperatureRecognition
from human_detection import HumanDetection
from waterlogging_detection import WaterloggingDetection
from data_process import ALDetector
import traceback

app = FastAPI()
# 设置允许访问的域名
origins = ["*"]  # 也可以设置为"*"，即为所有。

# 设置跨域传参
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  # 允许跨域的headers，可以用来鉴别来源等作用。

cache_dict = defaultdict(deque)
max_cache_length = 30


# ------ 获取当前时间下，摄像头获取的图像的燃气表数值 ------
@app.get("/recognition_dev", tags=["get current gas meter predict number"])
def pred_cur_num(item_id=None):
    """
    :param item_id: 燃气表编号
    :return:
    """

    # 设置参数
    result = None
    if item_id is not None and item_id in model_cfg.valid_number:
        request_params = {"equipId": item_id}
        res = requests.get(url=model_cfg.base64_url, params=request_params)
        try:
            b64_str = res.text
        except ValueError:
            print('出现{},摄像机的url存在问题！'.format(res.status_code))
            b64_str = None
        if b64_str is not None:
            img_string = base64.b64decode(b64_str)
            img_arr = np.frombuffer(img_string, np.uint8)
            image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            nr = NumberRecognition(model_cfg.model_path,
                                   model_cfg.gas_plate_model,
                                   model_cfg.gas_number_model)
            predict_df, status_code, predict_res = nr.predict(image)

            current_time = time.strftime('%Y-%m-%d %H:%M:%S')
            res = {
                "time": current_time,
                "status_code": status_code,
                "predict_number": predict_res
            }
            result = JSONResponse(status_code=200, content=res)
    if result is None:
        result = HTTPException(status_code=404, detail="燃气表编号无效或摄像头存在问题")
    return result


@app.post("/recognition", tags=["get current gas meter predict number"])
def pred_num(b64: str = Body(None, embed=True), data_type: str = Body('gas', embed=True),
             item_id: str = Body(None, embed=True), area_coordinate: list = Body(None, embed=True)):
    """
    燃气表|压力表读数识别
    :param b64: 图片base64地址
    :param data_type: 数据类型
    :param item_id: 图片标识，用于缓存队列校验使用
    :param area_coordinate: 表盘参考区域
    :return:
    """
    plate_model = model_cfg.gas_plate_model
    number_model = model_cfg.gas_number_model
    abnormal_save_folder = model_cfg.abnormal_save_folder_gas
    recognition_class = NumberRecognition
    area_coordinate_ = None
    if data_type == 'pressure':
        plate_model = model_cfg.pressure_plate_model
        number_model = model_cfg.pressure_pointer_model
        abnormal_save_folder = model_cfg.abnormal_save_folder_pressure
        recognition_class = PointerRecognition
        area_coordinate_ = area_coordinate
    try:
        img_string = base64.b64decode(b64)
        img_arr = np.frombuffer(img_string, np.uint8)
        # image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR to RGB
        # 最新模型输入是BGR, crop输出的图片是RGB
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        pr = recognition_class(model_cfg.model_path, plate_model, number_model)
        res_dict = pr.predict(image, area_coordinate_)
        plate_res, predict_df = res_dict['plate_res'], res_dict['predict_df']
        status_code, predict_res = res_dict['status_code'], res_dict['number_res']

        # 缓存历史值校验总量
        if data_type == 'gas' and item_id is not None:
            max_volume_per_hour = 100
            if item_id not in cache_dict and status_code == 0:
                cache_dict[item_id].append(float(predict_res[0]))
            elif item_id in cache_dict:
                first_read = cache_dict[item_id][0]
                before_read = cache_dict[item_id][-1]
                current_read = float(predict_res[0]) if 0 <= status_code <= 4 else -1
                if (current_read < max(first_read, before_read)
                        or current_read - first_read >= max_volume_per_hour) and status_code != 0:
                    add_num = np.diff(cache_dict[item_id]).mean()
                    if np.isnan(add_num) or add_num >= max_volume_per_hour / 2:
                        add_num = 0
                    current_read = before_read + add_num
                    if len(predict_res) > 0:
                        predict_res[0] = f'{current_read:.4f}'
                    else:
                        predict_res.append(f'{current_read:.4f}')
                    status_code = 10
                cache_dict[item_id].append(round(current_read, 4))
                if len(cache_dict[item_id]) > max_cache_length:
                    cache_dict[item_id].popleft()
            print('cache_dict: ', cache_dict)

        # status code大于0的存入abnormal_save_folder，后续需要人工重新标注
        if status_code > 0:
            abnormal_save_image_path = os.path.join(model_cfg.model_path, abnormal_save_folder, 'images')
            abnormal_save_label_path = os.path.join(model_cfg.model_path, abnormal_save_folder, 'labels')
            if not os.path.exists(abnormal_save_image_path):
                os.makedirs(abnormal_save_image_path)
            if not os.path.exists(abnormal_save_label_path):
                os.makedirs(abnormal_save_label_path)
            # 生成时间字符串作为图片名结尾
            image_suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-2]
            while os.path.exists(os.path.join(abnormal_save_image_path, f'img{image_suffix}.jpg')):
                time.sleep(np.random.rand())
                image_suffix = datetime.now().strftime('%Y%m%d%H%M%S%f')[:-2]

            # 检测表盘
            for idx, sub_plate in enumerate(plate_res):
                if sub_plate is not None and len(sub_plate) > 0:
                    cv2.imwrite(os.path.join(abnormal_save_image_path, f'img{image_suffix}.jpg'),
                                sub_plate[:, :, ::-1])

                    # 预测结果存储到txt文件
                    with open(os.path.join(abnormal_save_label_path, f'img{image_suffix}.txt'), 'w') as f:
                        for _, s in predict_df[idx].iterrows():
                            label_class = f'{s["class"]}'
                            label_xcenter = f'{s["xcenter"]:.6f}'
                            label_ycenter = f'{s["ycenter"]:.6f}'
                            label_width = f'{s["width"]:.6f}'
                            label_height = f'{s["height"]:.6f}'
                            label = ' '.join([label_class, label_xcenter,
                                              label_ycenter, label_width, label_height])
                            f.write(f'{label}\n')

        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        res = {
            "time": current_time,
            "status_code": status_code,
            "predict_number": predict_res
        }
        print(res)
        if data_type == 'gas':
            res['valid_flag'] = True if 0 <= status_code <= 4 or status_code == 10 else False
        elif data_type == 'pressure':
            res['labeled_image'] = res_dict['labeled_image']
            res['valid_flag'] = True if status_code == 0 else False
        result = JSONResponse(status_code=200, content=res)
    except ValueError:
        traceback.print_exc()
        result = HTTPException(status_code=404, detail="请输入有效的base64图片!")
    return result


@app.post("/temp", tags=["get temperature for object(i.e. pump)"])
def pred_temp(b64: str = Body(None, embed=True), item_id: str = Body(None, embed=True),
              area_coordinate: list = Body(None, embed=True)):
    """
    红外温度识别
    :param b64: 图片base64地址
    :param item_id: 图片标识
    :param area_coordinate: 目标区域坐标
    :return:
    """
    temp_model = model_cfg.temp_model
    try:
        img_string = base64.b64decode(b64)
        img_arr = np.frombuffer(img_string, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        tr = TemperatureRecognition(model_cfg.model_path, temp_model)
        predict_res, base64_img, status_code = tr.predict(image, item_id=item_id, area_coordinate=area_coordinate)

        current_time = time.strftime('%Y-%m-%d %H:%M:%S')
        res = {
            "time": current_time,
            'status_code': status_code,
            "predict_temp": predict_res,
            "labeled_image": base64_img
        }
        result = JSONResponse(status_code=200, content=res)
    except ValueError:
        traceback.print_exc()
        result = HTTPException(status_code=404, detail="请输入有效的base64图片!")
    return result


@app.post("/human", tags=["human recognition"])
def pred_human(b64: str = Body(None, embed=True)):
    """
    人员闯入图像识别，检测图片中是否有人(使用yolov8模型)
    :param b64: 图片base64地址
    :return:
    """
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        img_string = base64.b64decode(b64)
        img_arr = np.frombuffer(img_string, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        tr = HumanDetection(model_cfg.model_path)
        detection_res, status_code = tr.predict(image)

        res = {
            "time": current_time,
            'status_code': status_code,
            'detection_res': detection_res
        }
        print(res)
        result = JSONResponse(status_code=200, content=res)
    except ValueError:
        traceback.print_exc()
        res = {
            "time": current_time,
            'status_code': 1,
            'detection_res': False
        }
        result = JSONResponse(status_code=200, content=res)
    return result


@app.post("/waterlogging", tags=["waterlogging recognition"])
def pred_waterlogging(b64: str = Body(None, embed=True)):
    """
    水渍检测识别(分类模型，使用yolov8训练)
    :param b64: 图片base64地址
    :return:
    """
    current_time = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        img_string = base64.b64decode(b64)
        img_arr = np.frombuffer(img_string, np.uint8)
        image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        tr = WaterloggingDetection(model_cfg.model_path)
        detection_res, status_code = tr.predict(image)

        res = {
            "time": current_time,
            'status_code': status_code,
            'detection_res': detection_res
        }
        print(res)
        result = JSONResponse(status_code=200, content=res)
    except ValueError:
        traceback.print_exc()
        res = {
            "time": current_time,
            'status_code': 1,
            'detection_res': False
        }
        result = JSONResponse(status_code=200, content=res)
    return result


@app.post("/audio", tags=["get audio analysis"])
def audio(b64: str = Body(None, embed=True), data_type: str = Body('amplitude', embed=True)):
    """
    音频识别，返回声谱图和声压振幅曲线数据
    Args:
        b64: base64格式的图片数据
        data_type: amplitude | spectrogram

    Returns:

    """
    try:
        save_path = os.path.join('data', 'audio')
        temp_file = os.path.join(save_path, 'temp.WAV')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        wav_file = open(temp_file, "wb")
        decode_string = base64.b64decode(b64)
        wav_file.write(decode_string)
        wav_file.close()
        if data_type == 'amplitude':
            ar_obj = AudioRecognition(temp_file, sr=100)
            time_arr, audio_arr = ar_obj.get_amplitude_curve(list_format=True)
            res = {
                'time': time_arr,
                'amplitude': audio_arr
            }
        elif data_type == 'spectrogram':
            ar_obj = AudioRecognition(temp_file)
            freq_arr, time_arr, xdb, db_max = ar_obj.get_spectrogram(
                list_format=True, n_fft=512, hop_length=800)
            res = {
                'frequency': freq_arr,
                'time': time_arr,
                'spectrogram': xdb,
                'db_max': db_max
            }
        else:  # audio abnormal detection
            ar_obj = AudioRecognition(temp_file)
            abnormal_flag = ar_obj.abnormal_detect(model_cfg.audio_samples_path)
            res = {
                'abnormal_flag': abnormal_flag
            }
        result = JSONResponse(status_code=200, content=res)
    except ValueError:
        traceback.print_exc()
        result = HTTPException(status_code=404, detail="请输入有效的base64音频!")
    return result


if __name__ == '__main__':
    # pred_cur_num('474902489004576')
    ALDetector().run()
    uvicorn.run(app='main:app', host="0.0.0.0", port=8088)