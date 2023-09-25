import os
from ultralytics import YOLO


class WaterloggingDetection:

    def __init__(self, model_path, model_name='water-v1.pt'):
        self.model_path = os.path.join(model_path, 'models_pt', model_name)

    def predict(self, image):
        detection_res, status_code = False, 0
        try:
            if image is None:
                raise ValueError
            model = YOLO(self.model_path)
            results = model.predict(image, imgsz=416, augment=True)
            res = results[0].probs.top1
            res_str = 'have waterlogging!' if res == 1 else "don't have waterlogging"
            print('waterlogging prediction: ', res_str)
            if res == 1:
                detection_res = True
        except:
            status_code = 1
            pass

        return detection_res, status_code
