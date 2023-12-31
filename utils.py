import os
import pickle

import jams
import librosa
import pandas as pd
import sklearn

from tension_map import *


def preprocess_audio(audio_file, jam_file):
    with open('./src/crema/crema/models/chord/pump.pkl', 'rb') as fd:
        pump = pickle.load(fd)

    data = pump.transform(audio_file, jam_file)
    cqt = data['cqt/mag'].squeeze()

    return cqt


def prepare_data(database, meta_data, split, filter=None):
    features = []
    chords = []
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

            if filter:
                if quality not in filter:
                    continue

            # create input CQT features
            cqt = preprocess_audio(file_path, jam_path)
            features.append(cqt)
            nframe = cqt.shape[0]

            # create labels
            ori, tension = chord2polar(gt_chord)
            oris.append(np.full((nframe, 1), ori))
            tensions.append(np.array([tension] * nframe))
            chords.append(gt_chord)

    return (np.array(features), np.array(oris), np.array(tensions)), chords


def match_chord2jam(gt_chord):
    if "min7b5" in gt_chord:
        gt_chord = gt_chord.replace('min7b5', 'hdim7')
    if "seventh" in gt_chord:
        gt_chord = gt_chord.replace('seventh', '7')
    if 'sixth' in gt_chord:
        gt_chord = gt_chord.replace('sixth', 'maj6')

    return gt_chord


def create_jams(input_folder, output_folder):
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in [f for f in filenames if f.endswith(".wav")]:

            parts = filename.split('-')
            root = parts[0]
            quality = parts[2]
            gt_chord = match_chord2jam(root + ':' + quality)

            # ignore all the dyads
            if quality in ['min2', 'maj2', 'min3', 'maj3', 'perf4', 'tritone', 'perf5', 'min6', 'maj6',
                           'aug6', 'maj7_2', 'octave']:
                continue

            file_path = os.path.join(dirpath, filename)
            audio, sr = librosa.load(file_path)

            jam = jam_label(len(audio) / sr, gt_chord)
            jam.save(f'{output_folder}/{filename.split(".")[0]}.jams')


def chord_id(audio, sr, model):
    chord_est = model.predict(y=audio, sr=sr)
    jam = jams.JAMS()
    jam.file_metadata.duration = len(audio) / sr
    jam.annotations.append(chord_est)

    return chord_est.to_dataframe(), jam


def jam_label(duration, chord):
    jam = jams.JAMS()
    jam.file_metadata.duration = duration
    ann = jams.Annotation(namespace='chord', time=0, duration=jam.file_metadata.duration)
    ann.append(time=0, duration=duration, confidence=1, value=chord)
    jam.annotations.append(ann)

    return jam


def get_loader(database, metadata, split):
    meta = pd.read_csv(metadata)
    audio_ls = []
    sr_ls = []
    gt_chord_ls = []
    for index, row in meta[meta['split'] == split].iterrows():
        if 'chord' in row['mode']:
            labels = row['name'].split('-')
            folder = row['mode'][:-6]
            file_name = row['name']
            gt_chord = f"{labels[0]}:{folder}"

            file_path = f"{database}/{folder}/{file_name}.wav"
            audio, sr = librosa.load(file_path)

            audio_ls.append(audio)
            sr_ls.append(sr)
            gt_chord_ls.append(gt_chord)

    data = list(zip(audio_ls, sr_ls, gt_chord_ls))
    return data


def chord_acc(j_gt, j_est):
    results = {}
    for i, (gt, est) in enumerate(zip(j_gt, j_est)):
        ref_ann = gt.search(namespace='chord')[0]
        est_ann = est.search(namespace='chord')[0]
        results[i] = jams.eval.chord(est_ann, ref_ann)

    return pd.DataFrame(results)


def color_acc(t_gt, t_est):
    # mean square error of orientation and categorical error
    ori_gt = [x[0] for x in t_gt]
    tension_gt = [x[1] for x in t_gt]
    ori_est = [x[0] for x in t_est]
    tension_est = [x[1] for x in t_est]
    ori_mse = np.square(np.subtract(ori_gt, ori_est)).mean()

    tension_acc = sklearn.metrics.accuracy_score(np.array(tension_gt), np.array(tension_est))

    return ori_mse, tension_acc


if __name__ == '__main__':
    data_path = "/Users/sivanding/database/jazznet/chords"
    metadata = "/Users/sivanding/database/jazznet/metadata/chords"

    create_jams(data_path, metadata)
