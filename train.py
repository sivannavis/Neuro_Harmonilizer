"""
This script prepares data pumps for training and evaluation.

Author: Sivan Ding
sivan.d@nyu.edu

Reference:
    https://www.tensorflow.org/tutorials/keras/save_and_load
"""

from model import *
from utils import *

BATCH_SIZE = 5
EPOCHS = 100
HOP_LENGTH = 4096
SAMPLE_RATE = 44100


def train(model, train_data, val_data, output_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    feat, ori, tension = train_data
    val_ft, val_or, val_ten = val_data
    print(f'#training data: {feat.shape[0]}\n #validation data: {val_ft.shape[0]}\n')

    # save model weights every 5 epochs
    checkpoint_path = './models/' + output_dir + "/cp-{epoch:04d}.ckpt"
    cp_callback = k.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5 * int(np.ceil(feat.shape[0] / batch_size)))

    history = model.fit(feat, [ori, tension],
                        validation_data=(val_ft, [val_or, val_ten]),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[cp_callback],
                        verbose=2)
    print(history.history)

    # save model
    model.save(f'./models/best_model_{output_dir}.keras')

    # save model history
    with open(f'./models/history_{output_dir}', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return model


if __name__ == '__main__':
    # get datasets
    database = "/Users/sivanding/database/jazznet/chords"
    metadata = "/Users/sivanding/database/jazznet/metadata/tiny.csv"
    train_data, _ = prepare_data(database, metadata, 'train')
    val_data, _ = prepare_data(database, metadata, 'validation')
    test_data, _ = prepare_data(database, metadata, 'test')
    test_ft, test_or, test_ten = test_data

    # get model
    model = tension_model()

    # train and validation
    train(model, train_data, val_data, 'train_01')

    # testing
    print(f'#test data{test_ft.shape[0]}\n')
    scores = model.evaluate(test_ft, [test_or, test_ten])
    print(scores)
