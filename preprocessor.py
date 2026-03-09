"""
preprocessor.py
멜로디 데이터로부터 학습용 4D 텐서(X)와 레이블(Y)을 생성한다.
- create_curve_seq: 연속 노트 간 음정차·시간차·악기ID 시퀀스 생성
- get_data_set_of_XY: 전체 전처리 파이프라인 (MIDI → 정규화된 NumPy 배열)
"""

import os
import pickle
import logging

import numpy as np
from sklearn.preprocessing import normalize
from tensorflow.keras.utils import to_categorical

from config import (
    TRAIN_DATA_DIR,
    PREPROCESSED_DIR,
    PREDICT_PREPROCESSED_DIR,
    NUM_INSTRUMENTS,
    MAX_TIME_STEPS,
    TIME_RESOLUTION,
)
from data_loader import get_midi_set

logger = logging.getLogger(__name__)


def create_curve_seq(mel_arr):
    """연속된 노트 간의 [음정차, 시간차, 악기ID] 시퀀스를 반환한다."""
    curve_seq = []
    if len(mel_arr) > 1:
        for idx in range(1, len(mel_arr)):
            prev, curr = mel_arr[idx - 1], mel_arr[idx]
            if len(prev) == 3 and len(curr) == 3:
                pitch_diff = curr['note'] - prev['note']
                time_diff = curr['offset'] - prev['offset']
                instrument_id = 0 if prev['instrument'] is None else prev['instrument'] + 1
                curve_seq.append([pitch_diff, time_diff, instrument_id])
    return curve_seq


def get_data_set_of_XY(data_dir=None, predict=False):
    """
    학습 또는 예측에 사용할 (X, Y) 데이터셋을 반환한다.

    Parameters
    ----------
    data_dir : str, optional
        MIDI 파일이 있는 루트 디렉토리. None이면 TRAIN_DATA_DIR 사용.
    predict : bool
        True이면 레이블 없이 X만 반환한다.

    Returns
    -------
    arr1 : np.ndarray, shape (N, NUM_INSTRUMENTS, MAX_TIME_STEPS, 2)
    Y_set : np.ndarray or list
        학습 시 one-hot 레이블 배열, 예측 시 빈 리스트.
    """
    if data_dir is None:
        data_dir = TRAIN_DATA_DIR

    preprocessed_dir = PREDICT_PREPROCESSED_DIR if predict else PREPROCESSED_DIR

    # 장르 폴더 목록 수집
    if not predict:
        genre_names = sorted([
            name for name in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, name))
        ])
        paths = [os.path.join(data_dir, name) for name in genre_names]
    else:
        genre_names = []
        paths = [data_dir]

    class_num = len(paths)
    arr = []
    Y_set = []

    if not os.path.isdir(preprocessed_dir):
        logger.info("MIDI 파일을 새로 전처리합니다...")
        for i, path in enumerate(paths):
            name = genre_names[i] if not predict else os.path.basename(data_dir.rstrip(os.sep))
            songs = get_midi_set(path, name, predict)
            arr.append(songs)
            if not predict:
                for _ in songs:
                    Y_set.append(to_categorical(i, class_num))
    else:
        logger.info("전처리된 데이터 로드 중: %s", preprocessed_dir)
        for i, file_name in enumerate(sorted(os.listdir(preprocessed_dir))):
            if file_name.endswith('.p'):
                with open(os.path.join(preprocessed_dir, file_name), "rb") as fp:
                    songs = pickle.load(fp)
                arr.append(songs)
                if not predict:
                    for _ in songs:
                        Y_set.append(to_categorical(i, class_num))

    logger.info("2차 전처리: 미분 시퀀스 생성 중...")
    curve_seq_list = []
    raw_arr = []
    for mel_arr_list in arr:
        for mel_arr in mel_arr_list:
            curve_seq_list.append(create_curve_seq(mel_arr))
            raw_arr.append(mel_arr)
    del arr

    arr1 = np.zeros(
        (len(curve_seq_list), NUM_INSTRUMENTS, MAX_TIME_STEPS, 2),
        dtype=np.float32,
    )

    for i, seq in enumerate(curve_seq_list):
        for j, note_feat in enumerate(seq):
            t = int(raw_arr[i][j]['offset'] * TIME_RESOLUTION)
            inst_id = note_feat[2]
            if t < MAX_TIME_STEPS and inst_id != 0 and note_feat[1] != 0:
                arr1[i][inst_id][t] = note_feat[:2]  # [음정차, 시간차] 저장
    del raw_arr

    for i in range(len(arr1)):
        for j in range(len(arr1[i])):
            arr1[i][j] = normalize(arr1[i][j], axis=0)

    logger.info("입력 shape: %s", arr1.shape)

    if not predict:
        Y_set = np.reshape(np.array(Y_set), (-1, class_num))
        logger.info("출력 shape: %s", Y_set.shape)

    logger.info("전처리 완료.")
    return arr1, Y_set
