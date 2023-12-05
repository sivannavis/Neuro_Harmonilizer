"""
This script does chord recognition from music audio.

Author: Sivan Ding
sivan.d@nyu.edu

References:
    https://github.com/bmcfee/crema

"""
import numpy as np
from crema.analyze import analyze

a = np.zeros((2,4))
jam = analyze(filename='/Users/sivanding/database/jazznet/chords/sixth/G-2-sixth-chord-3.wav')
chords = jam.annotations['chord', 0]
chords.to_dataframe()


