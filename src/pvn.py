import os
import keras
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import regularizers
from keras.losses import categorical_crossentropy, mean_squared_error
from config import (
    BOARD_SHAPE,
    NUM_RESIDUAL_BLOCKS,
    NUM_NETWORK_UNITS,
    L2_PENALTY,
    BATCH_SIZE
)


class PolicyValueNet:
    def __init__(self, model_path):
        self.nn = self.build()
        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.load()

# Model utility
    def predict(self, features):
        return self.nn.predict(features, batch_size=BATCH_SIZE)

    def train(self, x, y, epochs=5, batch_size=BATCH_SIZE):
        self.nn.fit(x, y, epochs=epochs, batch_size=batch_size)

    def save(self, filepath=None):
        filepath = filepath or self.model_path
        self.nn.save_weights(filepath)

    def load(self, filepath=None):
        filepath = filepath or self.model_path
        self.nn.load_weights(filepath)
        print('pvn: Model loaded.')

# CNN
    def build_cnn(self):
        inputs = Input(shape=BOARD_SHAPE + (2,))
        nn = Conv2D(
            filters=32, kernel_size=3, padding='same', activation='relu'
        )(inputs)
        nn = Conv2D(
            filters=64, kernel_size=3, padding='same', activation='relu'
        )(nn)
        nn = MaxPooling2D(padding='same')(nn)
        nn = Flatten()(nn)
        nn = Dense(units=128, activation='relu')(nn)
        nn = Dropout(0.4)(nn)
        P = Dense(units=np.product(BOARD_SHAPE), activation='softmax')(nn)
        P = Reshape(BOARD_SHAPE, name='policy')(P)
        v = Dense(units=32, activation='relu')(nn)
        v = Dense(units=1, activation='tanh', name='value')(v)
        nn = Model(inputs=inputs, outputs=[P, v])
        nn.compile(
            optimizer='rmsprop',
            loss=self.loss,
            metrics=['accuracy']
        )
        return nn

# Network architecture
    def build(self):
        inputs = Input(shape=BOARD_SHAPE + (2,))
        nn = self.conv_block(
            inputs, filters=NUM_NETWORK_UNITS, kernel_size=3
        )
        for i in range(NUM_RESIDUAL_BLOCKS):
            nn = self.residual_block(nn)
        P = self.policy_head(nn)
        v = self.value_head(nn)
        nn = Model(inputs=inputs, outputs=[P, v])
        nn.compile(
            optimizer='rmsprop',
            loss=self.loss,
            metrics=['accuracy']
        )
        return nn

    def loss(self, y_true, y_pred):
        # It's the policy_loss if it has two dims.
        if y_pred.shape.as_list()[-2:] == list(BOARD_SHAPE):
            loss = categorical_crossentropy(y_true, y_pred)
        else:
            loss = mean_squared_error(y_true, y_pred)
        return loss

    def conv_block(self, x, filters, kernel_size):
        x = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=1,
            padding='same',
            kernel_regularizer=regularizers.l2(L2_PENALTY),
            bias_regularizer=regularizers.l2(L2_PENALTY)
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def residual_block(self, x):
        conv1 = self.conv_block(
            x, filters=NUM_NETWORK_UNITS, kernel_size=3
        )
        conv2 = self.conv_block(
            conv1, filters=NUM_NETWORK_UNITS, kernel_size=3
        )
        shortcut = Add()([conv2, x])
        return Activation('relu')(shortcut)

    def policy_head(self, x):
        x = self.conv_block(x, filters=2, kernel_size=1)
        x = Flatten()(x)
        x = Dense(
            units=np.product(BOARD_SHAPE),
            activation='softmax',
            kernel_regularizer=regularizers.l2(L2_PENALTY),
            bias_regularizer=regularizers.l2(L2_PENALTY)
        )(x)
        x = Reshape(BOARD_SHAPE, name='policy')(x)
        return x

    def value_head(self, x):
        x = self.conv_block(x, filters=1, kernel_size=1)
        x = Flatten()(x)
        x = Dense(
            units=NUM_NETWORK_UNITS,
            activation='relu',
            kernel_regularizer=regularizers.l2(L2_PENALTY),
            bias_regularizer=regularizers.l2(L2_PENALTY)
        )(x)
        x = Dense(
            units=1,
            activation='tanh',
            name='value',
            kernel_regularizer=regularizers.l2(L2_PENALTY),
            bias_regularizer=regularizers.l2(L2_PENALTY)
        )(x)
        return x
