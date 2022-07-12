import random
import pandas as pd
import numpy as np
import os
import librosa

from tqdm.auto import tqdm

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from audiomentations import Compose, TimeStretch, PitchShift, TimeMask, TanhDistortion, SpecCompose, SpecChannelShuffle

import warnings
warnings.filterwarnings(action='ignore')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


def get_mfcc_feature(df, data_type, save_path):
    # Data Folder path
    root_folder = 'D:/meta2/DACON/Corona'
    if os.path.exists(save_path):
        print(f'{save_path} is exist.')
        return
    features = []
    for uid in tqdm(df['id']):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')

        # librosa패키지를 사용하여 wav 파일 load
        y, sr = librosa.load(path, sr=CFG['SR'])

        # librosa패키지를 사용하여 mfcc 추출
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CFG['N_MFCC'])

        y_feature = []
        # 추출된 MFCC들의 평균을 Feature로 사용
        for e in mfcc:
            y_feature.append(np.mean(e))
        features.append(y_feature)

    # 기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
    mfcc_df = pd.DataFrame(features, columns=['mfcc_' + str(x) for x in range(1, CFG['N_MFCC'] + 1)])
    df = pd.concat([df, mfcc_df], axis=1)
    df.to_csv(save_path, index=False)
    print('Done.')

def onehot_encoding(ohe, x):
    # 학습데이터로 부터 fit된 one-hot encoder (ohe)를 받아 transform 시켜주는 함수
    encoded = ohe.transform(x['gender'].values.reshape(-1,1))
    encoded_df = pd.DataFrame(encoded, columns=ohe.categories_[0])
    x = pd.concat([x.drop(columns=['gender']), encoded_df], axis=1)
    return x

def param_loader(path, processor, max_seconds):
    wav, sfr = librosa.load(path, sr=16000)
    wav = wav.astype(np.float32)
    wav -= wav.mean()
    wav.resize(max_seconds*sfr)
    y = processor(wav, sampling_rate=sfr, return_tensors="np").input_values
    y = y.squeeze(0).astype(np.float32)
    return y


def make_dataset(main_path, data, idx, data_type, max_seconds, processor):
    root_folder = main_path
    features = []
    for uid in tqdm(idx):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')
        y_feature = param_loader(path, processor, max_seconds)
        features.append(y_feature)

    # 기존의 자가진단 정보를 담은 데이터프레임에 추출된 오디오 Feature를 추가
    features = np.stack(features, axis=0)
    dataset = np.concatenate([data, features], axis=1)
    return dataset

def simple_data(main_path, idx, data_type):
    root_folder = main_path
    features = []
    for uid in tqdm(idx):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')
        wav, sfr = librosa.load(path, sr=16000)
        wav = wav[:157440]
        if wav.shape[0] < 157440:
            remain = 157440 - wav.shape[0]
            wav = np.pad(wav, (0, remain), 'constant', constant_values=0)
        features.append(wav)
    return np.stack(features, axis=0)

def make_melspectrogram0(main_path, idx, data_type):
    root_folder = main_path
    features = []
    frame_length = 0.065
    frame_stride = 0.022
    frame_mel = 448
    for uid in tqdm(idx):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')
        wav, sfr = librosa.load(path, sr=16000)
        #157440 -> 448
        wav = wav[:157440]
        if wav.shape[0] < 157440:
            remain = 157440 - wav.shape[0]
            wav = np.pad(wav, (0,remain), 'constant', constant_values=0)
        input_nfft = int(round(sfr * frame_length))
        input_stride = int(round(sfr * frame_stride))
        S = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride)
        features.append(S)
    features = np.stack(features, axis=0)
    features = np.expand_dims(features, 3)
    return features


import torchaudio
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError

class AddGaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_amplitude=0.5, **kwargs):
        super().__init__(always_apply, p)
        self.noise_amplitude = (0.0, max_noise_amplitude)

    def apply(self, y: np.ndarray, **params):
        # 一様分布からサンプルを抽出
        noise_amplitude = np.random.uniform(*self.noise_amplitude)
        # 標準正規分布から出力値分をランダムで出力
        noise = np.random.randn(len(y))
        # 拡張
        augmented = (y + noise * noise_amplitude).astype(y.dtype)
        return augmented

class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)
        # 5
        self.min_snr = min_snr
        # 20
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented

import colorednoise as cn

class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented

def make_melspectrogram1(main_path, idx, data_type):
    root_folder = main_path
    features0 = [] #original
    features1 = [] #gaussian
    features2 = [] #pink
    transform0 = AddGaussianNoise(always_apply=True, max_noise_amplitude=0.05)
    transform1 = PinkNoiseSNR(always_apply=True, min_snr=5.0, max_snr=20.0)
    frame_length = 0.065
    frame_stride = 0.022
    frame_mel = 448
    for uid in tqdm(idx):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')
        wav, sfr = librosa.load(path, sr=16000)
        gaussian_wav = transform0(wav)
        pink_wav = transform1(wav)
        #157440 -> 448
        wav = wav[:157440]
        gaussian_wav = gaussian_wav[:157440]
        pink_wav = pink_wav[:157440]
        if wav.shape[0] < 157440:
            remain = 157440 - wav.shape[0]
            wav = np.pad(wav, (0,remain), 'constant', constant_values=0)
            gaussian_wav = np.pad(gaussian_wav, (0, remain), 'constant', constant_values=0)
            pink_wav = np.pad(pink_wav, (0, remain), 'constant', constant_values=0)

        input_nfft = int(round(sfr * frame_length))
        input_stride = int(round(sfr * frame_stride))
        S0_0 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride)
        S0_1 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                              win_length=input_nfft//2)
        S0_2 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                              win_length=input_nfft//4)
        S1_0 = librosa.feature.melspectrogram(y=gaussian_wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride)
        S1_1 = librosa.feature.melspectrogram(y=gaussian_wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                            win_length=input_nfft//2)
        S1_2 = librosa.feature.melspectrogram(y=gaussian_wav, n_mels=frame_mel, n_fft=input_nfft,
                                              hop_length=input_stride, win_length=input_nfft//4)
        S2_0 = librosa.feature.melspectrogram(y=pink_wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride)
        S2_1 = librosa.feature.melspectrogram(y=pink_wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                              win_length=input_nfft//2)
        S2_2 = librosa.feature.melspectrogram(y=pink_wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                              win_length=input_nfft//4)
        S0 = np.stack([S0_0, S0_1, S0_2], axis=2)
        S1 = np.stack([S1_0, S1_1, S1_2], axis=2)
        S2 = np.stack([S2_0, S2_1, S2_2], axis=2)
        features0.append(S0)
        features1.append(S1)
        features2.append(S2)
    features0 = np.stack(features0, axis=0)
    features1 = np.stack(features1, axis=0)
    features2 = np.stack(features2, axis=0)

    return features0, features1, features2

def stack_mel(wav, frame_mel, input_nfft, input_stride):
    S0_0 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride)
    S0_1 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                          win_length=input_nfft // 2)
    S0_2 = librosa.feature.melspectrogram(y=wav, n_mels=frame_mel, n_fft=input_nfft, hop_length=input_stride, \
                                          win_length=input_nfft // 4)
    S0 = np.stack([S0_0, S0_1, S0_2], axis=2)
    return S0

#augmentation
def make_melspectrogram2(main_path, idx, data_type):
    root_folder = main_path
    aug0 = Compose([TimeStretch(min_rate=0.8, max_rate=1.25, p=1)])
    aug1 = Compose([PitchShift(min_semitones=-4, max_semitones=4, p=1)])
    aug2 = Compose([TimeMask(p=1)])
    aug3 = Compose([TanhDistortion(p=1)])
    features0 = [] #original
    features1 = [] #gaussian
    features2 = [] #pink
    features3 = []
    features4 = []
    features5 = []
    features6 = []
    transform0 = AddGaussianNoise(always_apply=True, max_noise_amplitude=0.05)
    transform1 = PinkNoiseSNR(always_apply=True, min_snr=5.0, max_snr=20.0)
    frame_length = 0.065
    frame_stride = 0.022
    frame_mel = 448
    for uid in tqdm(idx):
        root_path = os.path.join(root_folder, data_type)
        path = os.path.join(root_path, str(uid).zfill(5) + '.wav')
        wav, sfr = librosa.load(path, sr=16000)
        gaussian_wav = transform0(wav)
        pink_wav = transform1(wav)
        #157440 -> 448
        wav = wav[:157440]
        gaussian_wav = gaussian_wav[:157440]
        pink_wav = pink_wav[:157440]
        if wav.shape[0] < 157440:
            remain = 157440 - wav.shape[0]
            wav = np.pad(wav, (0,remain), 'constant', constant_values=0)
            gaussian_wav = np.pad(gaussian_wav, (0, remain), 'constant', constant_values=0)
            pink_wav = np.pad(pink_wav, (0, remain), 'constant', constant_values=0)

        wav0 = aug0(wav, sample_rate=16000)
        wav1 = aug1(wav, sample_rate=16000)
        wav2 = aug2(wav, sample_rate=16000)
        wav3 = aug3(wav, sample_rate=16000)

        input_nfft = int(round(sfr * frame_length))
        input_stride = int(round(sfr * frame_stride))
        S0 = stack_mel(wav, frame_mel, input_nfft, input_stride)
        S1 = stack_mel(gaussian_wav, frame_mel, input_nfft, input_stride)
        S2 = stack_mel(pink_wav, frame_mel, input_nfft, input_stride)
        S3 = stack_mel(wav0, frame_mel, input_nfft, input_stride)
        S4 = stack_mel(wav1, frame_mel, input_nfft, input_stride)
        S5 = stack_mel(wav2, frame_mel, input_nfft, input_stride)
        S6 = stack_mel(wav3, frame_mel, input_nfft, input_stride)


        features0.append(S0)
        features1.append(S1)
        features2.append(S2)
        features3.append(S3)
        features4.append(S4)
        features5.append(S5)
        features6.append(S6)

    features0 = np.stack(features0, axis=0)
    features1 = np.stack(features1, axis=0)
    features2 = np.stack(features2, axis=0)
    features3 = np.stack(features3, axis=0)
    features4 = np.stack(features4, axis=0)
    features5 = np.stack(features5, axis=0)
    features6 = np.stack(features6, axis=0)

    return features0, features1, features2, features3, features4, features5, features6

# transform = AddGaussianNoise(always_apply=True, max_noise_amplitude=0.05)
# y_gaussian_added = transform(y)
#
# transform = GaussianNoiseSNR(always_apply=True, min_snr=5, max_snr=20)
# y_gaussian_snr = transform(y)
#
# transform = PinkNoiseSNR(always_apply=True, min_snr=5.0, max_snr=20.0)
# y_pink_noise = transform(twav)
import torch
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam