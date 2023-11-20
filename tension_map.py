'''
This script maps chord to tension quality values.

Author: Sivan Ding
sivan.d@nyu.edu
'''
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

def chord2notes(chord_name):
    notes = []
    return notes

def note2ori(note):
    orientation = []
    return orientation

def notes2ten(notes):
    tension = []
    return tension