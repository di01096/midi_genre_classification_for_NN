"""
model.py
MIDI 장르 분류를 위한 CNN 기반 신경망 모델 정의.
"""

from tensorflow.keras import layers, models, Input


class NN(models.Sequential):
    """
    3개의 Conv2D 블록으로 구성된 장르 분류 CNN.

    Parameters
    ----------
    input_shape_4d : tuple
        (batch, instruments, timesteps, features) 형태의 입력 shape.
    num_classes : int
        분류할 장르 수.
    """

    def __init__(self, input_shape_4d, num_classes):
        super().__init__()
        feature_shape = (input_shape_4d[1], input_shape_4d[2], input_shape_4d[3])

        self.add(Input(shape=feature_shape))

        # Conv Block 1
        self.add(layers.Conv2D(8, kernel_size=(3, 3), activation='relu',
                               padding='same'))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())

        # Conv Block 2
        self.add(layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())

        # Conv Block 3
        self.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
        self.add(layers.Dropout(0.5))
        self.add(layers.BatchNormalization())
        self.add(layers.MaxPool2D())

        # Classifier
        self.add(layers.Flatten())
        self.add(layers.Dense(64, activation='relu'))
        self.add(layers.Dropout(0.2))
        self.add(layers.BatchNormalization())
        self.add(layers.Dense(num_classes, activation='softmax'))

        self.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )
