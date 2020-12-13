# In[]
import os
import numpy as np
import pandas as pd

import soundfile as sf

# Bandpass
from scipy import signal

import matplotlib.pyplot as plt
import librosa
import librosa.display

from python_speech_features import mfcc

# Classifier
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_curve, auc, plot_confusion_matrix

from tqdm import tqdm #PROGRESS BAR

# In[]
# =============================================================================
# READ 'patient_diagnosis.csv'
# =============================================================================
def ReadCSV(path_csv):
    df = pd.read_csv(path_csv, names = ['ID', 'Diagnosis'])
    df.loc[df['Diagnosis'] != 'COPD', ['Diagnosis']] = 0
    df.loc[df['Diagnosis'] == 'COPD', ['Diagnosis']] = 1
    df.set_index('ID', inplace = True)
    print(df)
    return df

# In[]
# =============================================================================
# READ audio_and_txt_files
# =============================================================================
def ReadWAV(path_wav, df):
    signals = []
    sample_rates = []
    sig_names = []
    labels = []
    
    files = os.listdir(path_wav)
    
    progress = tqdm(total = len(files)) #PROGRESS BAR
    for file in files:
        progress.update(1) #PROGRESS BAR
        
        if file.endswith("wav"):
            signal, sample_rate = sf.read(path_wav + '/' + file)
            signal = signal[0:int(10.0 * sample_rate)] # Keep the first 20 seconds
            label = df.loc[int(file[0:3])]['Diagnosis']
            
            sig_resize = np.resize(signal, (441000, ))
                
            signals.append(sig_resize)
            sample_rates.append(sample_rate)
            sig_names.append(file)
            labels.append(label)
            
        # if file.endswith("txt"):
        #     pass
        
    progress.close() #PROGRESS BAR
    return signals, sample_rates, sig_names, labels

# In[]
# =============================================================================
# PRINT WAV AMPLITUDE
# =============================================================================
def PrintAMP(signal, name):
    plt.figure(figsize = (10,5))
    plt.ylabel('Amplitude')
    plt.title(name)
    librosa.display.waveplot(signal, sr=44100)
    plt.xlabel('Time [sec]')
    plt.show()

# In[]
# =============================================================================
# PRINT WAV FREQUENCY
# =============================================================================
def PrintFREQ(signal, sr, name):
    s = librosa.stft(signal)
    s_db = librosa.amplitude_to_db(abs(s))
    
    plt.figure(figsize=(10,5))
    librosa.display.specshow(s_db, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar(format="%+2.f dB")
    plt.title(name)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

# In[]
# =============================================================================
# MFCC
# =============================================================================
def MFCC(signals, sample_rates):
    mfcc_sigs = []
    i = 0
    
    for signal in signals:
        mfcc_sig = mfcc(signal, sample_rates[i], nfft=1103)
        mfcc_reshape = np.reshape(np.resize(mfcc_sig, (1999, 13)), -1)

        mfcc_sigs.append(mfcc_reshape)
        
    return mfcc_sigs  
        

# In[]
# =============================================================================
# Bandpass
# =============================================================================
def Bandpass(signals):
    filtered_sigs = []
    i = 0
    for sig in signals:
        b, a = signal.butter(8, [0.2,0.8], 'bandpass')
        filtered_sig = signal.filtfilt(b, a, sig)
        
        filtered_sigs.append(filtered_sig)
    return filtered_sigs

# In[]
# =============================================================================
# Classifier SVM
# =============================================================================
def ClassifySVM(train_sig, train_labels, test_sig, test_labels):
    print("\n==================================================================");
    SVM = svm.SVC()
    svm_fit = SVM.fit(train_sig, train_labels)
    svm_labels = SVM.predict(test_sig)
    
    accuracy_svm = accuracy_score(test_labels, svm_labels)
    precision_svm = precision_score(test_labels, svm_labels)
    recall_svm = recall_score(test_labels, svm_labels)
    print("For Support Vector Machine:")
    print("    Accuracy: ", '%.3f'%accuracy_svm)
    print("    Precision: ", '%.3f'%precision_svm)
    print("    Recall: ", '%.3f'%recall_svm)
    
    TN, FP, FN, TP = confusion_matrix(test_labels, svm_labels).ravel()
    print("    Sensitivity: ", '%.3f'%(TP / (TP + FN)))
    print("    Specificity: ", '%.3f'%(TN / (TN + FP)))
        
    fpr, tpr, _ = roc_curve(test_labels, svm_labels)
    roc_auc = auc(fpr, tpr)
    print("    AUC: ", '%.3f'%roc_auc)
    
    plot_confusion_matrix(SVM, test_sig, test_labels, cmap=plt.cm.Blues) 
    plt.show()
    

# In[]
# =============================================================================
# Classifier Random Forest
# =============================================================================
def ClassifyRF(train_sig, train_labels, test_sig, test_labels):
    print("\n==================================================================");
    RF = RandomForestClassifier(n_estimators = 500)
    RF_fit = RF.fit(train_sig, train_labels)
    RF_labels = RF_fit.predict(test_sig)
    
    accuracy_RF = accuracy_score(test_labels, RF_labels)
    precision_RF = precision_score(test_labels, RF_labels)
    recall_RF = recall_score(test_labels, RF_labels)
    print("For Random Forest Classifier:")
    print("    Acurracy: ", '%.3f'%accuracy_RF)
    print("    Precision: ", '%.3f'%precision_RF)
    print("    Recall: ", '%.3f'%recall_RF)
    
    TN, FP, FN, TP = confusion_matrix(test_labels, RF_labels).ravel()
    print("    Sensitivity: ", '%.3f'%(TP / (TP + FN)))
    print("    Specificity: ", '%.3f'%(TN / (TN + FP)))
        
    fpr, tpr, _ = roc_curve(test_labels, RF_labels)
    roc_auc = auc(fpr, tpr)
    print("    AUC: ", '%.3f'%roc_auc)
    
    plot_confusion_matrix(RF, test_sig, test_labels, cmap=plt.cm.Blues) 
    plt.show()
    

# In[]
df_csv = ReadCSV('C:/STAY/專題/respiratory analysis/Datasets/Respiratory_Sound_Database/patient_diagnosis.csv')

# In[]
path = 'C:/STAY/專題/respiratory analysis/Datasets/Respiratory_Sound_Database/audio_and_txt_files'
signals, sample_rates, sig_names, labels = ReadWAV(path, df_csv)

# PrintAMP(signals[0], sig_names[0])
# PrintFREQ(signals[0], 44100, sig_names[0])
# PrintAMP(signals[4], sig_names[4])
# PrintFREQ(signals[4], 44100, sig_names[4])



# In[]
# =============================================================================
# Original
# =============================================================================
print("\n**************************** Original ****************************")
x_train, x_test, y_train, y_test = train_test_split(signals, labels,
                                                    test_size=0.3, random_state=None)

ClassifySVM(x_train, y_train, x_test, y_test)
ClassifyRF(x_train, y_train, x_test, y_test)


# In[]
print("\n****************************** MFCC ******************************")
mfcc_sigs = MFCC(signals, sample_rates);

# PrintAMP(mfcc_sigs[0], sig_names[0])
# PrintFREQ(mfcc_sigs[0], sample_rates[0], sig_names[0])
# PrintAMP(mfcc_sigs[4], sig_names[4])
# PrintFREQ(mfcc_sigs[4], 44100, sig_names[4])

x_train, x_test, y_train, y_test = train_test_split(mfcc_sigs, labels,
                                                    test_size=0.3, random_state=None)
ClassifySVM(x_train, y_train, x_test, y_test)
ClassifyRF(x_train, y_train, x_test, y_test)

# In[]
# =============================================================================
# Bandpass
# =============================================================================
print("\n**************************** Bandpass ****************************")
filtered_sigs = Bandpass(signals)

# PrintAMP(filtered_sigs[0], sig_names[0])
# PrintFREQ(filtered_sigs[0], sample_rates[0], sig_names[0])
# PrintAMP(filtered_sigs[4], sig_names[4])
# PrintFREQ(filtered_sigs[4], 44100, sig_names[4])

x_train, x_test, y_train, y_test = train_test_split(filtered_sigs, labels,
                                                    test_size=0.3, random_state=None)
ClassifySVM(x_train, y_train, x_test, y_test)
ClassifyRF(x_train, y_train, x_test, y_test)


# In[]
print("\n************************* Bandpass + MFCC *************************")
mfcc_sigs = MFCC(filtered_sigs, sample_rates);

# PrintAMP(mfcc_sigs[0], sig_names[0])
# PrintFREQ(mfcc_sigs[0], sample_rates[0], sig_names[0])
# PrintAMP(mfcc_sigs[4], sig_names[4])
# PrintFREQ(mfcc_sigs[4], 44100, sig_names[4])

x_train, x_test, y_train, y_test = train_test_split(mfcc_sigs, labels,
                                                    test_size=0.3, random_state=None)
ClassifySVM(x_train, y_train, x_test, y_test)
ClassifyRF(x_train, y_train, x_test, y_test)

