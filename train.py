"""
train.py
CNN 모델을 학습하고 가중치와 학습 곡선 그래프를 저장한다.

Usage:
    python train.py
"""

import logging

import matplotlib.pyplot as plt

from config import MODEL_PATH, EPOCHS, BATCH_SIZE, VALIDATION_SPLIT
from preprocessor import get_data_set_of_XY
from model import NN
from visualizer import plot_acc, plot_loss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)
logger = logging.getLogger(__name__)


def main():
    logger.info("학습 데이터 로드 중...")
    X_train, Y_train = get_data_set_of_XY()

    num_classes = Y_train.shape[1]
    model = NN(X_train.shape, num_classes)
    model.summary()

    logger.info("모델 학습 시작 (epochs=%d, batch_size=%d)...", EPOCHS, BATCH_SIZE)
    history = model.fit(
        X_train, Y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )

    model.save_weights(MODEL_PATH)
    logger.info("모델 저장 완료: %s", MODEL_PATH)

    plot_loss(history, title='Training Loss')
    plt.savefig('train_loss.png')
    plt.clf()

    plot_acc(history, title='Training Accuracy')
    plt.savefig('train_acc.png')
    plt.clf()

    logger.info("그래프 저장 완료 (train_loss.png, train_acc.png)")


if __name__ == '__main__':
    main()
