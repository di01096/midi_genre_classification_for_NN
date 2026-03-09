"""
predict.py
학습된 모델로 MIDI 파일의 장르를 예측하고 확률을 출력한다.

Usage:
    python predict.py
"""

import os
import logging

import numpy as np

from config import MODEL_PATH, PREDICT_DATA_DIR, TRAIN_DATA_DIR
from preprocessor import get_data_set_of_XY
from model import NN

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    genre_names = sorted([
        name for name in os.listdir(TRAIN_DATA_DIR)
        if os.path.isdir(os.path.join(TRAIN_DATA_DIR, name))
    ])
    num_classes = len(genre_names)

    logger.info("예측 데이터 전처리 중...")
    X, _ = get_data_set_of_XY(data_dir=PREDICT_DATA_DIR, predict=True)

    model = NN(X.shape, num_classes)
    model.load_weights(MODEL_PATH)

    predict_files = sorted(os.listdir(PREDICT_DATA_DIR))
    for i, arr in enumerate(X):
        arr_input = np.reshape(arr, (1, *arr.shape))
        score = model.predict(arr_input, verbose=0)
        print(f"\n[{predict_files[i]}]")
        for j, genre in enumerate(genre_names):
            print(f"  {genre}: {score[0][j]:.4f}")


if __name__ == '__main__':
    main()
