# Music_Harmonilizer
recognizing harmony qualities from music audio

Author: Sivan Ding, Vio Chung, Rave Rajan

## What's a harmonilizer?
It maps any chord to a polar coordinates $\phi$, $\rho$, where $\phi$ means the color orientation and the $\rho$ means the tension class within the total 31 classes.

## How does it work?
We modified a chord recognition model to be a tension embedding extractor, then feed it into a MLP to do regression on chord 
orientation $\phi$ and categorical classification on tension $\rho$ at the same time with a combination of 
MSE loss and categorical crossentropy loss.

### To run the baseline: tension identifier
1. Initialize a chord recognition model from `crema`
2. Get chord and tension metrics

### To run our method: neuro-harmonilizer
Please follow `./Notebook/demo.ipynb`
1. Initialize a fixed and non-fixed tension model
2. Train and validate both tension models
3. Evaluate both tension models
4. Run the training process diagnostics 

### To run the analysis
Please follow `./Notebook/analysis.ipynb`
1. Create model architecture and load model weights for both fixed and non-fixed tension model
2. Compare models through spectrograms, forward and backward GRU, and cqt
3. Show model results of fixed vs. unfixed models
4. Observe the performance of the neuro-harmonilizer in individual chords using `metric_filter` function
5. Observe the performance of the neuro-harmonilizer in triads vs. tetrads
6. Observe the performance of indidvidual tension class

## But does it actually work?
Yes, it does! We compared a naive mapping baseline and our modified neural network based method and it shows some advantages.
The experiments are done using JazzNet, a dataset that contains chords/arpeggio/scales independent piano audio. 

Baseline: Chord recognition -> map chord directly to harmony colors

Ours: Chord embedding extractor -> classifier -> harmony colors

## What's it for?
Higher level musical quality extraction. It is intended to use for controllable music audio data analysis and generation.

