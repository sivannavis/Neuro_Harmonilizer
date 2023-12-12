"""
This script prepares data pumps for training and evaluation.
"""
import os.path

from model import *
from tension_map import *
from utils import *

BATCH_SIZE = 1
EPOCHS = 1
HOP_LENGTH = 4096
SAMPLE_RATE = 44100


def preprocess_audio(audio_file, jam_file):
    with open('./src/crema/crema/models/chord/pump.pkl', 'rb') as fd:
        pump = pickle.load(fd)

    data = pump.transform(audio_file, jam_file)
    cqt = data['cqt/mag'].squeeze()

    return cqt


def prepare_data(database, meta_data, split):
    features = []
    oris = []
    tensions = []
    meta = pd.read_csv(meta_data)

    for index, row in meta[meta['split'] == split].iterrows():
        if 'chord' in row['mode']:
            folder = row['mode'][:-6]
            file_name = row['name']
            parts = file_name.split('-')
            root = parts[0]
            quality = parts[2]
            gt_chord = match_chord2jam(root + ':' + quality)
            file_path = f"{database}/{folder}/{file_name}.wav"
            jam_path = f'{os.path.dirname(meta_data)}/chords/{file_name.split(".")[0]}.jams'

            # ignore all the dyads
            if quality in ['min2', 'maj2', 'min3', 'maj3', 'perf4', 'tritone', 'perf5', 'min6', 'maj6',
                           'aug6', 'maj7_2', 'octave']:
                continue

            # create input CQT features
            cqt = preprocess_audio(file_path, jam_path)
            features.append(cqt)
            nframe = cqt.shape[0]

            # create labels
            ori, tension = chord2polar(gt_chord)
            oris.append(np.full((nframe, 1), ori))
            tensions.append(np.array([tension] * nframe))
    return (np.array(features), np.array(oris), np.array(tensions))


def train(model, train_data, val_data):
    feat, ori, tension = train_data
    val_ft, val_or, val_ten = val_data
    history = model.fit(feat, [ori, tension], validation_data=(val_ft, [val_or, val_ten]),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2)
    print(history.history)
    return model


def root(x):
    return os.path.splitext(os.path.basename(x))[0]


if __name__ == '__main__':
    # get datasets
    database = "/Users/sivanding/database/jazznet/chords"
    metadata = "/Users/sivanding/database/jazznet/metadata/test.csv"
    train_data = prepare_data(database, metadata, 'train')
    val_data = prepare_data(database, metadata, 'validation')
    test_ft, test_or, test_ten = prepare_data(database, metadata, 'test')

    # get model
    model = tension_model()

    # train and validation
    train(model, train_data, val_data)

    # testing
    scores = model.evaluate(test_ft, [test_or, test_ten])
    print(scores)
