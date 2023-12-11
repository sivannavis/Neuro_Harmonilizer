"""
This script does chord recognition from music audio.

Author: Sivan Ding
sivan.d@nyu.edu

References:
    https://github.com/bmcfee/crema
    https://github.com/ejhumphrey/ace-lessons
    https://github.com/bmcfee/ismir2017_chords

"""

import pickle

import crema
from tension_map import *
from utils import *


def loader_id(loader, model):
    chord_gt = []
    chord_est = []
    color_gt = []
    color_est = []
    jam_gt = []
    jam_est = []

    for audio, sr, gt_chord in loader:
        # let's first get rid of dyads...
        if gt_chord.split(':')[1] in ['min2', 'maj2', 'min3', 'maj3', 'perf4', 'tritone', 'perf5', 'min6', 'maj6',
                                      'aug6', 'maj7_2', 'octave']:
            continue

        if "min7b5" in gt_chord:
            gt_chord = gt_chord.replace('min7b5', 'hdim7')
        if "seventh" in gt_chord:
            gt_chord = gt_chord.replace('seventh', '7')
        if 'sixth' in gt_chord:
            gt_chord = gt_chord.replace('sixth', 'maj6')
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


if __name__ == '__main__':
    # let's consider jazznet format only for now.
    # custom configurations
    database = "/Users/sivanding/database/jazznet/chords"
    metadata = "/Users/sivanding/database/jazznet/metadata/tiny.csv"
    # get model
    model = crema.models.chord.ChordModel()

    # load dataset
    test_loader = get_loader(database, metadata, split='test')
    with open("test_loader", "wb") as fp:  # Pickling
        pickle.dump(test_loader, fp)
    with open("test_loader", "rb") as fp:  # Unpickling
        test_loader = pickle.load(fp)

    # get results
    c_gt, c_est, t_gt, t_est, j_gt, j_est = loader_id(test_loader, model)

    # get metrics
    chord_met = chord_acc(j_gt, j_est)
    tension_met = color_acc(t_gt, t_est)

    print(chord_met.mean(axis=1))
    print(tension_met)
