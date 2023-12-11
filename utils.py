import os

import jams
import librosa
import numpy as np
import pandas as pd
import sklearn


def create_jams(input_folder, output_folder):
    for dirpath, dirnames, filenames in os.walk(input_folder):
        for filename in [f for f in filenames if f.endswith(".wav")]:
            file_path = os.path.join(dirpath, filename)


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
