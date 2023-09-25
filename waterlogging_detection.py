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
            results = model.predict(image, classes=0)
            print('waterlogging detection boxes: ')
            res = results[0].boxes
            print(res)
            if res.shape[0] > 0:
                detection_res = True
        except:
            status_code = 1
            pass

        return detection_res, status_code
