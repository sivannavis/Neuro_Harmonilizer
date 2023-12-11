"""
This script adapts crema model into pytorch model, outputs the latent features and feeds into a naive classifier.

Author: Sivan Ding
sivan.d@nyu.edu

References:
    https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/
    https://github.com/szagoruyko/functional-zoo/blob/master/resnet-18-export.ipynb
    https://keras.io/guides/transfer_learning/

"""

import pickle

import tensorflow.keras as k
from tensorflow.keras.models import model_from_config
from tensorflow.keras.utils import plot_model

from crema import layers


class Tension(k.Model):
    def __init__(self):
        super().__init__()

        # fix layers in crema model
        # self.input = k.layers.Input(batch_input_shape=(216, 2))
        self.b1 = k.layers.BatchNormalization()
        self.b1.trainable = False

        self.c0 = k.layers.Convolution2D(1, (5, 5), padding='same',
                                         activation='relu',
                                         data_format='channels_last')
        self.c0.trainable = False

        self.b2 = k.layers.BatchNormalization()
        self.b2.trainable = False

        self.c1 = k.layers.Convolution2D(36, (1, 216), padding='valid', activation='relu',
                                         data_format='channels_last')
        self.c1.trainable = False

        self.b3 = k.layers.BatchNormalization()
        self.b3.trainable = False

        self.r1 = k.layers.Lambda(lambda x: k.backend.squeeze(x, axis=2))
        self.r1.trainable = False

        self.rs = k.layers.Bidirectional(k.layers.GRU(64, return_sequences=True))
        self.rs.trainable = False

        self.b4 = k.layers.BatchNormalization()
        self.b4.trainable = False

        self.rs2 = k.layers.Bidirectional(k.layers.GRU(64, return_sequences=True))
        self.rs2.trainable = False

        self.b5 = k.layers.BatchNormalization()
        self.b5.trainable = False

        self.classifier = k.Sequential(
            k.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
            k.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
        )

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

    def get_weights(self, weights_path):
        pass


if __name__ == '__main__':
    with open('/Users/sivanding/Codebase/Neuro_Harmonilizer/src/crema/crema/models/chord/pump.pkl', 'rb') as fd:
        pump = pickle.load(fd)

    print(pump.fields)

    LAYERS = pump['cqt'].layers()

    x = LAYERS['cqt/mag']

    # Now load the model
    with open('/Users/sivanding/Codebase/Neuro_Harmonilizer/src/crema/crema/models/chord/model_spec.pkl',
              'rb') as fd:
        spec = pickle.load(fd)
        model = model_from_config(spec,
                                  custom_objects={k: layers.__dict__[k]
                                                  for k in layers.__all__})

    # And the model weights
    model.load_weights('/Users/sivanding/Codebase/Neuro_Harmonilizer/src/crema/crema/models/chord/model.h5',
                       )

    tension = Tension()

    # plot_model(model, to_file='model.png', show_shapes=True)

    for index in range(1, 10):
        extracted_weights = model.layers[index].get_weights()
        tension.layers[index - 1].set_weights(extracted_weights)
