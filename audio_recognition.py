import numpy as np
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
sns.set_style("darkgrid")


class AudioRecognition:

    def __init__(self, audio_path, sr=None):
        self.audio_path = audio_path
        self.audio, self.sr = librosa.load(audio_path, sr=sr)

    def get_amplitude_curve(self, plot=False, save_path=None, list_format=False, q=4):
        if plot:
            plt.figure(figsize=(15, 10))
            librosa.display.waveshow(self.audio, sr=self.sr)
        if save_path is not None:
            father_folder = os.path.join(*re.split(r'[\\/]', save_path)[:-1])
            if not os.path.exists(father_folder):
                os.makedirs(father_folder)
            plt.savefig(save_path)
        time_arr = np.array([i / self.sr for i in np.arange(len(self.audio))])
        audio = self.audio
        print('amplitude shape: ', audio.shape)
        if list_format:
            time_arr = [float(f'{x:.{q-1}f}') for x in time_arr]
            audio = [float(f'{x:.{q}f}') for x in audio]
            # 时间补整
            total_time = len(audio) / self.sr
            time_arr.append(total_time)
            audio.append(audio[-1])
        return time_arr, audio

    def get_spectrogram(self, n_fft=1024, win_length=None, hop_length=None,
                        plot=False, save_path=None, list_format=False, q=4):
        if win_length is None:
            win_length = n_fft
        if hop_length is None:
            hop_length = win_length // 2
        x = librosa.stft(self.audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        xdb = librosa.amplitude_to_db(abs(x))  # X--二维数组数据
        if plot:
            plt.figure(figsize=(14, 5))
            librosa.display.specshow(xdb, sr=self.sr, hop_length=hop_length, x_axis='time', y_axis='hz')
            plt.colorbar(format="%+2.0f dB")
            plt.title('STFT transform processing audio signal to spectrogram')
        if save_path is not None:
            father_folder = os.path.join(*re.split(r'[\\/]', save_path)[:-1])
            if not os.path.exists(father_folder):
                os.makedirs(father_folder)
            plt.savefig(save_path)

        # 转换为标准输出格式
        # 获取采集频率
        freq_arr = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft)
        # 获取时间序列
        total_points = xdb.shape[1] * hop_length
        total_time = total_points / self.sr
        idx_arr = librosa.frames_to_samples(np.arange(xdb.shape[1]), hop_length=hop_length)
        time_arr = np.array([i / total_points * total_time for i in idx_arr])
        print('xdb shape: ', xdb.shape)
        if list_format:
            xdb = [[float(f'{j:.{q}f}') for j in i] for i in xdb]
            freq_arr = [float(f'{i:.{q-1}f}') for i in freq_arr]
            time_arr = [float(f'{i:.{q-1}f}') for i in time_arr]
        return freq_arr, time_arr, xdb
