# COPD Analysis
### by Tzu Yi Chang, Shih Ya Wong, Cho Ting Lee.
#### This code is for "Respiratory Sound Analysis of Chronic Obstruction Pulmonary Disease".

* The dataset was created by two research teams in Portugal and Greece. It contains 920 recordings of varying length (10s to 90s) from 126 patients.
793 audio files are from patients with COPD, and the other 127 files are from healthy people or patients with other diseases.

* The code can be broken to several parts:
  * Data Preprocessing:
    * Take the first 20 seconds of every audio file.
    * For some datasets, apply bandpass filter.
    * Resize every data to the same size.

  * Split 70% as training dataset and 30% as testing dataset.

  * For some datasets, calculate MFFC.

  * Classification: 
    * Use SVM and Random Forest to classify
      * original data
      * original data through calculating MFCC
      * original data applying bandpass filter
      * original data with bandpass filter and through calculating MFCC
	
* Comparing the results, we can find that using data with bandpass filter and MFCC has the best performance.
