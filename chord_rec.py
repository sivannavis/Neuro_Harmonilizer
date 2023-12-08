"""
This script does chord recognition from music audio.

Author: Sivan Ding
sivan.d@nyu.edu

References:
    https://github.com/bmcfee/crema
    https://github.com/ejhumphrey/ace-lessons
    https://github.com/bmcfee/ismir2017_chords

"""

import jams
import librosa
import numpy as np
import pandas as pd
import sklearn
import pickle

import crema
from tension_map import *


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


def loader_id(loader, model):
    chord_gt = []
    chord_est = []
    color_gt = []
    color_est = []
    jam_gt = []
    jam_est = []

    for audio, sr, gt_chord in loader:
        # if 'octave' not in gt_chord:
        #     continue
        gt_color = chord2polar(gt_chord)
        preds, pred_jam = chord_id(audio, sr, model)
        pred_chord = preds['value'][0]
        pred_color = chord2polar(pred_chord)

        chord_gt.append(gt_chord)
        chord_est.append(pred_chord)
        color_gt.append(gt_color)
        color_est.append(pred_color)
        jam_gt.append(jam_label(len(audio) / sr, gt_chord))
        jam_est.append(pred_jam)

    return chord_gt, chord_est, color_gt, color_est, jam_gt, jam_est


def get_loader(databse, metadata, split):
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
    # let's consider jazznet format only for now.
    # custom configurations
    database = "/Users/sivanding/database/jazznet/chords"
    metadata = "/Users/sivanding/database/jazznet/metadata/tiny.csv"
    # get model
    model = crema.models.chord.ChordModel()

    # load dataset
    # test_loader = get_loader(database, metadata, split='test')
    # with open("test_loader", "wb") as fp:  # Pickling
    #     pickle.dump(test_loader, fp)
    with open("test_loader", "rb") as fp:  # Unpickling
        test_loader = pickle.load(fp)

    # get results
    c_gt, c_est, t_gt, t_est, j_gt, j_est = loader_id(test_loader, model)

    # get metrics
    chord_met = chord_acc(j_gt, j_est)
    tension_met = color_acc(t_gt, t_est)

    print(chord_met.mean(axis=1))
