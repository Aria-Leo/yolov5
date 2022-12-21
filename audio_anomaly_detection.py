
import os
import re
import glob
import shutil
import numpy as np
import librosa
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


class AudioAnomalyDetection:

    def __init__(self):
        self.files_train = None
        self.files_predict = None
        self.lof = None
        self.iforest = None
        self.pca = None

    def extract_features(self, folder_path, valid_format=('wav',),
                         max_counts=None, usage_flag=0):
        files = np.random.permutation(glob.glob(os.path.join(folder_path, '*')))
        if usage_flag == 0:
            self.files_train = files
        else:
            self.files_predict = files
        features = []
        if max_counts is None:
            max_counts = len(files)
        for f in files[:max_counts]:
            f_name = re.split(r'[/\\]', f)[-1]
            suffix = f_name.split('.')[-1].lower()
            if suffix not in valid_format:
                continue
            features.append(self.extract_single(f))
        features = np.array(features)
        return features

    @staticmethod
    def extract_single(audio_file, sr=None,
                       n_mfcc=40, hop_length=256, window='hamming'):
        x, sr = librosa.load(audio_file, sr=sr)
        mfccs = librosa.feature.mfcc(
            y=x, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, window=window)
        feature = np.mean(mfccs, axis=1)
        return feature

    def fit(self, features):
        # pca降维
        self.pca = PCA(n_components=features.shape[1] // 2)
        x = self.pca.fit_transform(features)
        self.iforest = IsolationForest(
            n_estimators=150, max_samples=0.25, max_features=0.9, random_state=666)
        self.iforest.fit(x)
        self.lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        self.lof.fit(features)
        return self

    def predict(self, features):
        x = self.pca.transform(features)
        result_iforest = self.iforest.predict(x)
        result_lof = self.lof.predict(features)
        abnormal_idx = np.argwhere((result_iforest == -1) & (result_lof == -1)).flatten()
        result = np.ones_like(result_iforest, dtype=int)
        result[abnormal_idx] = -1
        return result

    def fit_predict(self, features):
        self.fit(features)
        result = self.predict(features)
        return result

    def store(self, result, abnormal_path, usage_flag=0):
        if usage_flag == 0:
            abnormal_files = self.files_train[result == -1]
        else:
            abnormal_files = self.files_predict[result == -1]
        if not os.path.exists(abnormal_path):
            os.makedirs(abnormal_path)
        for af in abnormal_files:
            f_name = re.split(r'[/\\]', af)[-1]
            shutil.copyfile(af, os.path.join(abnormal_path, f_name))


if __name__ == '__main__':
    data_path = '/Users/ywtai/Desktop/工作/中瑞恒/jupyter/data/AudioData/raw_image'
    out_path = '/Users/ywtai/Desktop/工作/中瑞恒/jupyter/data/AudioData/abnormal_image_5'
    aad = AudioAnomalyDetection()
    fe = aad.extract_features(data_path)
    predict = aad.fit_predict(fe)
    aad.store(predict, out_path)
