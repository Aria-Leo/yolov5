import os
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils.metrics_extra import nms
import torch


class TemperatureRecognition:

    def __init__(self, model_path, model_name, conf=0.5, iou=0.25):
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

        self.device_reflect = {
            '1001': '预热机1-冷却水泵',
            '1002': '预热机2-冷却水泵',
            '1003': '预热机2-空调水泵',
            '1004': '预热机1-冷却水泵电机',
            '1005': '直燃机1-空调水泵',
            '1006': '直燃机2-冷却水泵',
            '1007': '直燃机1-冷却水泵',
            '1008': '电制冷冷却水-冷却水泵',
            '1009': '电制冷空调水1-冷却水泵',
            '1010': '电制冷空调水2-冷却水泵'
        }

        self.area_reflect = {
            '1001': [[0.443570, 0.429156, 0.163484, 0.483979]],
            '1002': [[0.450859, 0.449167, 0.141563, 0.456250], [0.622070, 0.548750, 0.163984, 0.528125]],
            '1003': [[0.450852, 0.388047, 0.199516, 0.562031]],
            '1004': [[0.672398, 0.428411, 0.180516, 0.589635]],
            '1005': [[0.596113, 0.341797, 0.199430, 0.486198]],
            '1006': [[0.094773, 0.416354, 0.185922, 0.634167], [0.469203, 0.385448, 0.168781, 0.405687]],
            '1007': [[0.342074, 0.377255, 0.184133, 0.359844]],
            '1008': [[0.435039, 0.386354, 0.151641, 0.370625], [0.588008, 0.400740, 0.147109, 0.317688]],
            '1009': [[0.078949, 0.260510, 0.157883, 0.513854], [0.273035, 0.280979, 0.197680, 0.456125]],
            '1010': [[0.109059, 0.224271, 0.214227, 0.439167], [0.364652, 0.233411, 0.226883, 0.432052]],
            'temp_bar': [0.980, 0.494, 0.035, 0.72]
        }

    def get_area(self, item_id=None):
        return self.area_reflect.get(item_id, [])

    @staticmethod
    def cv2_img_add_text(img, text, left, top, textColor=(0, 255, 0), textSize=20):
        if isinstance(img, np.ndarray):  # 判断是否OpenCV图片类型
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 创建一个可以在给定图像上绘图的对象
        draw = ImageDraw.Draw(img)
        # 字体的格式
        fontStyle = ImageFont.truetype(
            "fonts/simsun.ttc", textSize, encoding="utf-8")
        # 绘制文本
        draw.text((left, top), text, textColor, font=fontStyle)
        # 转换回OpenCV格式
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    def predict(self, image, item_id=None, area_coordinate=None, inference_size=960):
        result = self.model(image, size=inference_size, augment=True)
        predict_df = result.pandas().xywhn[0]
        predict_df = nms(predict_df)
        print(predict_df)
        temp_symbol = predict_df[predict_df['name'] == 'temp_symbol']
        if temp_symbol.shape[0] != 2:
            print('温度标识符号识别错误，无法判断温度颜色区间！')
            return None, 1

        need_clusters = predict_df[predict_df['name'] != 'temp_bar']

        features = np.array([[s['xcenter'], (s['ycenter'] + s['height'] / 2) * 100]
                             for _, s in need_clusters.iterrows()])
        choose_clusters, choose_ss = 1, 0  # 通过轮廓系数选定最优的聚类数
        for c in range(2, min(features.shape[0] - 1, 11)):
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
        temp_min, temp_max, temp_bar_arr = None, None, None
        for i in range(choose_clusters):
            ln_idx = np.argwhere(pre == i).flatten()
            # 寻找temp_symbol所在的温度上下限读数
            temp_df = need_clusters.iloc[ln_idx]
            if not temp_df[temp_df['name'] == 'temp_symbol'].empty:
                temp_df = temp_df[temp_df['name'] != 'temp_symbol']
                temp_df = temp_df.sort_values(by=['xcenter'])
                temp_number = float(''.join(temp_df['name'].tolist()))
                if temp_min is None:
                    temp_min = temp_number
                else:
                    if temp_number < temp_min:
                        temp_max = temp_min
                        temp_min = temp_number
                    else:
                        temp_max = temp_number
                    break
        # 存储temp_bar的区域
        print('temp range: ')
        print(temp_min, temp_max)
        temp_bar_s = self.area_reflect['temp_bar']
        x_range = [temp_bar_s[0] - temp_bar_s[2] / 2,
                   temp_bar_s[0] + temp_bar_s[2] / 2]
        y_range = [temp_bar_s[1] - temp_bar_s[3] / 2,
                   temp_bar_s[1] + temp_bar_s[3] / 2]
        x_range = [int(x * image.shape[1]) for x in x_range]
        y_range = [int(y * image.shape[0]) for y in y_range]
        temp_bar_arr = image[y_range[0]: y_range[1], x_range[0]: x_range[1], :]
        temp_res, base64_str = [], None
        if temp_min is not None and temp_max is not None and temp_bar_arr is not None:
            # 识别目标区域
            if area_coordinate is not None:
                area_reflect = area_coordinate
            else:
                area_reflect = self.get_area(item_id)

            cv_image = image.copy()
            for area in area_reflect:
                x_range = [area[0] - area[2] / 2,
                           area[0] + area[2] / 2]
                y_range = [area[1] - area[3] / 2,
                           area[1] + area[3] / 2]
                x_range = [int(x * image.shape[1]) for x in x_range]
                y_range = [int(y * image.shape[0]) for y in y_range]

                area_arr = image[y_range[0]: y_range[1], x_range[0]: x_range[1], :]
                current_res = self.cal_temp(temp_min, temp_max, temp_bar_arr, area_arr)
                temp_res.append(current_res)
                print(image.shape)
                print(x_range)
                print(y_range)

                # 返回标注图片
                cv_image = cv2.rectangle(cv_image, (x_range[0], y_range[0]),
                                         (x_range[1], y_range[1]), (0, 0, 255), 2)
                read_w = int(area[2] * image.shape[1])
                read_h = int(area[3] * image.shape[0])
                cv_image = self.cv2_img_add_text(cv_image, f'{current_res:.2f}\u2103',
                                                 x_range[0] + int(read_w * 0.1),
                                                 y_range[0] + int(read_h * 0.1),
                                                 textColor=(255, 255, 255),
                                                 textSize=max(int(read_w * 0.2), 20))
                # cv2.putText(cv_image, f'{current_res:.2f}C', (x_range[0] + 40, y_range[0] + 40),
                #             cv2.FONT_HERSHEY_PLAIN, 2.0, (255, 255, 255), 2)
            # image to base64
            base64_img = cv2.imencode('.jpg', cv_image)[1].tostring()
            base64_img = base64.b64encode(base64_img)
            base64_str = str(base64_img, 'utf-8')

        status_code = self.check_result(temp_res)
        return temp_res, base64_str, status_code

    @staticmethod
    def check_result(temp_res):
        status_code = 0
        if temp_res is None:
            status_code = 1
        elif not temp_res:
            status_code = 2
        return status_code

    @staticmethod
    def cal_temp(min_scale, max_scale, bar, target_area):
        """
        根据给定的红外温度上下限值，和温度条颜色矩阵，输出指定区域的颜色
        Args:
            min_scale: float
            max_scale: float
            bar: array, shape: (h, w, 3)
            target_area: array, shape: (h, w, 3)

        Returns:

        """
        # 温度等分为h份
        split_counts = bar.shape[0]
        t_scales = np.linspace(min_scale, max_scale, split_counts)
        features = bar.mean(axis=1)
        labels = t_scales
        # ridge回归
        alphas = np.linspace(0.01, 0.5, 100)
        ridge_model = RidgeCV(alphas=alphas)
        ridge_model.fit(features, labels)
        target_features = target_area.reshape(-1, 3)
        target_temperature = ridge_model.predict(target_features)
        res = min(np.percentile(target_temperature, 99.5), max_scale)
        print('predict temp: ', res)
        return res
