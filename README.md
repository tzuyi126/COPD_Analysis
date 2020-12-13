# COPD Analysis
## This code is for "Respiratory Sound Analysis of Chronic Obstruction Pulmonary Disease".

* The dataset was created by two research teams in Portugal and Greece. It contains 920 recordings of varying length (10s to 90s) from 126 patients.
793 audio files are from patients with COPD, and the other 127 files are from healthy people or patients with other diseases.

* The code can be broken to several parts:
  * Data Preprocessing:
    * Take the first 20 seconds of every audio file, then apply bandpass filter on and resize every data to the same size.

  * Split 70% as training data set and 30% as testing data set.

  * Calculate MFFC of every data

  * Classification: 
	Use SVM and Random Forest to classify (1)original data, (2)original data applying bandpass filter, (3)original data calculating MFCC, and (4)data with bandpass filter and * calculated MFCC
	
* Comparing the results, we can find that using data with bandpass filter and MFCC has the best achievement
