"""
This script adapts crema model into pytorch model, outputs the latent features and feeds into a naive classifier.

Author: Sivan Ding
sivan.d@nyu.edu

References:
    https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/
    https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb
    https://keras.io/guides/transfer_learning/
    https://stackoverflow.com/questions/43294367/how-can-i-load-the-weights-only-for-some-layers
    https://keras.io/api/models/model/
    https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
    https://keras.io/guides/training_with_built_in_methods/

"""

import pickle

import numpy as np
import tensorflow.keras as k
from tensorflow.keras.models import model_from_config
from tensorflow.keras.utils import plot_model

from crema import layers


class Tension(k.Model):  # TODO: abandoned for now, do not touch
    def __init__(self):
        super().__init__()
        self.crema_model = get_weights()

        # previous model architecture
        self.b1 = k.layers.BatchNormalization()
        self.b1.trainable = False

        self.c0 = k.layers.Convolution2D(1, (5, 5), padding='same',
                                         activation='relu',
                                         data_format='channels_last')
        self.c0.trainable = False

        self.b2 = k.layers.BatchNormalization()
        self.b2.trainable = False

        self.c1 = k.layers.Convolution2D(72, (1, 216), padding='valid', activation='relu',
                                         data_format='channels_last')
        self.c1.trainable = False

        self.b3 = k.layers.BatchNormalization()
        self.b3.trainable = False

        self.r1 = k.layers.Lambda(lambda x: k.backend.squeeze(x, axis=2))
        self.r1.trainable = False

        self.rs = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True), input_shape=(None, 72))
        self.rs.trainable = False

        self.b4 = k.layers.BatchNormalization()
        self.b4.trainable = False

        self.rs2 = k.layers.Bidirectional(k.layers.GRU(256, return_sequences=True), input_shape=(None, 256))
        self.rs2.trainable = False

        self.b5 = k.layers.BatchNormalization()
        self.b5.trainable = False

        self.classifier = k.Sequential([
            k.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
            k.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
        ])

        self.out1 = k.layers.Dense(1, activation='linear')
        self.out2 = k.layers.Dense(32, activation='softmax')

        self.time1 = k.layers.TimeDistributed(self.out1)
        self.time2 = k.layers.TimeDistributed(self.out2)

    def forward(self, inputs):
        # from input to embeddings
        x = self.b1(inputs)
        x = self.c0(x)
        x = self.b2(x)
        x = self.c1(x)
        x = self.b3(x)
        x = self.r1(x)
        x = self.rs(x)
        x = self.b4(x)
        hidden = self.rs2(x)

        # from latent features to numerical and categorical outputs
        x = self.b5(hidden)
        x = self.classifier(x)
        ori = self.time1(x)
        tension = self.time2(x)

        return hidden, ori, tension

    def call(self, inputs):
        hidden, ori, tension = self.forward(inputs)

        return ori, tension

    # def build(self, input_shapes):
    #     super(k.Model, self).build(input_shapes)


def get_weights():
    # Now load the model
    with open('./src/crema/crema/models/chord/model_spec.pkl',
              'rb') as fd:
        spec = pickle.load(fd)
        model = model_from_config(spec,
                                  custom_objects={k: layers.__dict__[k]
                                                  for k in layers.__all__})

    # And the model weights
    model.load_weights('./src/crema/crema/models/chord/model.h5',
                       )

    # plot_model(model, to_file='model.png', show_shapes=True)
    return model


def prepare_model():
    # get trained model
    tension_model = Tension()
    trained = get_weights()
    input_data = np.zeros((1, 32, 216, 2))
    output = tension_model(input_data)
    # tension_model.build((1, 32, 216, 2))
    tension_model.compile(loss=['mse', 'sparse_categorical_crossentropy'], optimizer='adam')
    # tension_model.build((1, 32, 216, 2))
    for index in range(1, 10):
        extracted_weights = trained.layers[index].get_weights()
        tension_model.layers[index - 1].set_weights(extracted_weights)

    # compile the keras model
    plot_model(tension_model, to_file='tension_model.png', show_shapes=True)

    return tension_model


def tension_model():
    trained = get_weights()
    input = trained.layers[0].input
    output = trained.layers[9].output
    classifier = k.Sequential([
        k.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        k.layers.Dense(128, activation='relu', kernel_initializer='he_normal'),
    ])(output)

    ori = k.layers.Dense(1, activation='linear')
    tension = k.layers.Dense(31, activation='softmax')

    ori = k.layers.TimeDistributed(ori)(classifier)
    tension = k.layers.TimeDistributed(tension)(classifier)

    model = k.models.Model(inputs=input, outputs=[ori, tension])

    plot_model(model, to_file='model.png', show_shapes=True)

    model.compile(loss=['mse', 'categorical_crossentropy'],
                  optimizer='adam',
                  metrics=['mean_squared_error', "categorical_accuracy"])

    return model


if __name__ == '__main__':
    with open('./src/crema/crema/models/chord/pump.pkl', 'rb') as fd:
        pump = pickle.load(fd)

    print(pump.fields)

    model = tension_model()
