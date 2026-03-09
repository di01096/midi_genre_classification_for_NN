"""
visualizer.py
학습 과정의 정확도·손실 곡선을 그래프로 시각화한다.
"""

import matplotlib.pyplot as plt


def plot_acc(history, title=None):
    """학습/검증 정확도 곡선을 그린다."""
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    if title:
        plt.title(title)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)


def plot_loss(history, title=None):
    """학습/검증 손실 곡선을 그린다."""
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    if title:
        plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc=0)
