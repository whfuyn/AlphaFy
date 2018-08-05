import os
import keras
import numpy as np
from keras import regularizers
from keras.models import Model, Input, load_model
from keras.losses import categorical_crossentropy, mean_squared_error
from keras.layers import (
    Add, Dense, Conv2D, Reshape, Flatten, BatchNormalization, Activation
)
from config import (
    BOARD_SHAPE,
    NUM_RESIDUAL_BLOCKS,
    NUM_NETWORK_UNITS,
    L2_PENALTY,
)


class PolicyValueNet:
    def __init__(self, model_path=None, config=dict()):
        self.num_blocks = config.get('num_blocks', NUM_RESIDUAL_BLOCKS)
        self.num_units = config.get('num_units', NUM_NETWORK_UNITS)
        self.l2_penalty = config.get('l2_penalty', L2_PENALTY)
        if model_path is None:
            self.nn = self.build()
            print("pvn: New model created.")
        elif os.path.exists(model_path):
            # Need to pass loss function through custom_objects.
            # I am not sure if I do it correctly, but it works.
            self.nn = load_model(model_path,
                                 custom_objects={'loss': self.loss})
            print("pvn: Model loaded. Path: \"{}\"".format(model_path))
        else:
            raise RuntimeError("pvn: Model file is not found.")

# Model utility
    def predict(self, features):
        return self.nn.predict(features)

    def train(self, x, y, epochs=5, batch_size=256):
        self.nn.fit(x, y, epochs=epochs, batch_size=batch_size)

    def save(self, path):
        self.nn.save(path)
        print("pvn: Model saved. Path: \"{}\"".format(path))

# Network architecture
    def build(self):
        inputs = Input(shape=BOARD_SHAPE + (2,))
        nn = self.conv_block(
            inputs, filters=self.num_units, kernel_size=3
        )
        for i in range(self.num_blocks):
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
            kernel_regularizer=regularizers.l2(self.l2_penalty),
            bias_regularizer=regularizers.l2(self.l2_penalty)
        )(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def residual_block(self, x):
        conv1 = self.conv_block(
            x, filters=self.num_units, kernel_size=3
        )
        conv2 = self.conv_block(
            conv1, filters=self.num_units, kernel_size=3
        )
        shortcut = Add()([conv2, x])
        return Activation('relu')(shortcut)

    def policy_head(self, x):
        x = self.conv_block(x, filters=2, kernel_size=1)
        x = Flatten()(x)
        x = Dense(
            units=np.product(BOARD_SHAPE),
            activation='softmax',
            kernel_regularizer=regularizers.l2(self.l2_penalty),
            bias_regularizer=regularizers.l2(self.l2_penalty)
        )(x)
        x = Reshape(BOARD_SHAPE, name='policy')(x)
        return x

    def value_head(self, x):
        x = self.conv_block(x, filters=1, kernel_size=1)
        x = Flatten()(x)
        x = Dense(
            units=self.num_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_penalty),
            bias_regularizer=regularizers.l2(self.l2_penalty)
        )(x)
        x = Dense(
            units=1,
            activation='tanh',
            name='value',
            kernel_regularizer=regularizers.l2(self.l2_penalty),
            bias_regularizer=regularizers.l2(self.l2_penalty)
        )(x)
        return x
