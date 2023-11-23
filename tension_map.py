"""
This script maps chord to tension quality values.
JazzNet: https://github.com/tosiron/jazznet

Author: Sivan Ding
sivan.d@nyu.edu
"""
import mirdata
import numpy as np


def chord2polar(chord_name):
    # orientation of chord
    notes = chord2notes(chord_name)
    orientations = []
    for n in notes:
        orientations.append(note2ori(n))
    orientation = np.mean(orientations)
    tension = notes2ten(notes)
    return orientation, tension


def chord2notes(chord_name):  # TODO
    notes = []
    return notes


def note2ori(note):
    """
    The polar coordinate of a note in circle of 5th. We use all flat notation and degree.
    Circle of 5th always rotates clockwise independent of start note.
    :param note:
    :return:
    """
    # notes = ['Eb', 'Ab', 'Db', 'Gb', 'B', 'E', 'A', 'D', 'G', 'C', 'F', 'Bb']
    # notes = ['C', 'G', 'D', 'A', 'E', 'B', 'Gb', 'Db', 'Ab', 'Eb', 'Bb', 'F']
    orients = {}
    orientation = 0
    for n in notes:
        orients[n] = orientation
        orientation += 30

    return orients[note]


def notes2ten(notes):
    """
    Assuming no duplicated notes are presented.
    :param notes:
    :return: tension level of the given sequence of notes.
    """
    n_notes = len(notes)
    assert n_notes > 1
    # calculate all angles
    angles = get_angles(notes)
    flat_angles = sum(angles, [])
    # calculate how many intervals in circle of 5th
    int_5 = max(flat_angles) / 30
    # calculate how many semitones
    int_min2 = sum([flat_angles == 150])
    # calculate how many major 2nd intervals
    int_maj2 = sum([flat_angles == 60])
    # calculate how many major 3rd
    int_maj3, int_min3 = check_maj3(angles)
    cont_semi = check_semi(angles)
    tension = get_tension(int_5, int_min2, int_maj2, int_min3, int_maj3, cont_semi)
    return tension


def get_tension(fifth, min_2, maj_2, min_3, maj_3, cont_semi):
    tension = np.zeros([30])
    if fifth == 1:
        raise NotImplementedError
    # level 0 to 2
    if fifth in [2, 3, 4] and min_2 == 0:
        if maj_2 <= 1:
            if min_3 or maj_3:
                tension[0] = 1
            else:
                tension[1] = 1
        elif maj_2 in [2, 3]:
            tension[2] = 1
    # level 3 to 5
    if fifth == 5 and min_2 == 1:
        if maj_3 or min_3:
            if maj_2 <= 1:
                tension[3] = 1
            elif maj_2 == 2:
                tension[4] = 1
            elif maj_2 > 2:
                tension[5] = 1
        else:
            tension[5] = 1
    if fifth == 6:
        # level 6 to 8
        if min_2 == 0:
            if maj_2 <= 1:
                if min_3 or maj_3:
                    tension[6] = 1
                else:
                    tension[7] = 1
            elif maj_2 == 3:
                tension[8] = 1
            else:
                raise NotImplementedError

        # level 9 to 11
        elif min_2 == 1:
            if maj_3 or min_3:
                if maj_2 == 1:
                    tension[9] = 1
                elif maj_2 == 2:
                    tension[10] = 1
                elif maj_2 > 2:
                    tension[11] = 1
            else:
                tension[11] = 1

        # level 12 to 13
        elif min_2 == 2:
            if maj_2 == 1 and (maj_3 or min_3):
                tension[12] = 1
            elif maj_2 > 1 or not (maj_3 or min_3):
                tension[13] = 1
        else:
            raise NotImplementedError

    if fifth in [7, 8, 9, 10, 11]:
        # level 14 to 16
        if min_2 == 0:
            if fifth == 8:
                if maj_2 == 0:
                    tension[14] = 1
                elif maj_2 == 2:
                    tension[15] = 1
            elif fifth > 8 or maj_2 > 2:
                tension[16] = 1

        # level 17 to 19
        elif min_2 == 1:
            if maj_2 == 0 and (maj_3 or min_3):
                tension[17] = 1
            elif maj_2 <= 2 and (maj_3 or min_3):
                tension[18] = 1
            else:
                tension[19] = 1

        # level 20 to 23
        elif min_2 == 2:
            if simple:
                if maj_2 <= 2 and (maj_3 or min_3):
                    tension[20] = 1
                else:
                    tension[21] = 1
            else:
                if maj_2 <= 3 and (maj_3 or min_3):
                    tension[22] = 1
                else:
                    tension[23] = 1

        # level 24 to 26
        elif min_2 == 3:
            if cont_semi[0]:
                tension[24] = 1
            elif cont_semi[1]:
                tension[25] = 1
            elif cont_semi[2]:
                tension[26] = 1
            else:
                raise NotImplementedError
        # level 27 to 29
        else:
            if cont_semi[1] == 2:
                tension[27] = 1
            elif cont_semi[2]:
                tension[28] = 1
            elif cont_semi[3]:
                tension[29] = 1
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    return tension


def check_maj3(angles):
    maj_3 = 0
    min_3 = 0
    for pairs in angles:
        for index, pair in enumerate(pairs):
            if pair == 120:
                if index != 0:
                    if angles[index - 1] == 30:
                        maj_3 += 1
                    elif angles[index - 1] == 90:
                        min_3 += 1
    return maj_3, min_3


def check_semi(angles):  # TODO
    """
    :param angles:
    :return: a list of how many groups of 1 semitone, 2-4 neighbor semitones.
    """
    cont_semi = [0, 0, 0, 0] # single, double, triple， quadruple

    # get connected pairs
    connects = []
    for i, pairs in enumerate(angles):
        for j, pair in enumerate(pairs):
            if pair == 150:
                connects.append((i, j + (i + 1)))

    # calculate connected components
    doubles = get_pairs(connects[1:], connects[0])
    triples = get_pairs(doubles[1:], doubles[0])
    quadruples = get_pairs(triples[1:], triples[0])

    cont_semi[0] = len(connects)
    cont_semi[1] = len(doubles)
    cont_semi[2] = len(triples)
    cont_semi[3] = len(quadruples)

    return cont_semi


def get_pairs(connects, target_pair):
    pairs = []
    if len(connects) == 1:
        if set(connects) & set(target_pair):  # if they have common elements in t uples
            pairs.append(tuple({*connects, *target_pair}))
    elif len(connects) > 1:
        for index, connect in enumerate(connects):
            pairs.append(*get_pairs(connect, target_pair))
        pairs.append(*get_pairs(connects[1:], connects[0]))
    else:
        raise ValueError("no connected pairs available")

    return pairs


def angle(ori1, ori2):
    if abs(ori1 - ori2) > 180:
        return 360 - abs(ori1 - ori2)
    else:
        return abs(ori1 - ori2)


def get_angles(notes):
    angles = []
    for index, i in enumerate(notes[:-1]):
        angles.append([])
        for j in notes[index + 1:]:
            if angle(i, j):
                angles[index].append(angle(i, j))
    return angles


if __name__ == '__main__':
    data = mirdata.initialize('guitarset', data_home='/Users/sivanding/database/guitarset')
