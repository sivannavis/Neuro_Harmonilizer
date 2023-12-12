# Music_Harmonilizer
recognizing harmony qualities from music audio

Author: Sivan Ding, Vio Chung, Rave Rajan

## What's a harmonilizer?
It maps any chord to a polar coordinates $$ \phi, \rho$$, where $$\phi$$ means the color orientation and the $$\rho$$ means the tension class within the total 31 classes.

## How does it work?
We modified a chord recognition model to be a tension embedding extractor, then feed it into a MLP to do regression on chord 
orientation $$\phi$$ and categorical classification on tension $$\rho$$ at the same time with a combination of 
MSE loss and categorical crossentropy loss.

## But does it actually work?
Yes, it does! We compared a naive mapping baseline and our modified neural network based method and it shows some advantages.
The experiments are done using JazzNet, a dataset that contains chords/arpeggio/scales independent piano audio. 

Baseline: Chord recognition -> map chord directly to harmony colors

Ours: Chord embedding extractor -> classifier -> harmony colors

## What's it for?
Higher level musical quality extraction. It is intended to use for controllable music audio data analysis and generation.

