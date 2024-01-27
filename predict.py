# -*- coding: utf-8 -*-
"""

Skript testet das vortrainierte Modell


@author:  Team Jumpman(Yubao Ma, Huanzheng Zhu)
"""

import os
import mne
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from tensorflow.keras.models import load_model
from CSP import CSPMF



###Signatur der Methode (Parameter und Anzahl return-Werte) darf nicht verändert werden
def predict_labels(channels : List[str], data : np.ndarray, fs : float, reference_system: str,
                    W1_name : str='model/W_vis5.npy', W2_name : str='model/W_vis4.npy',
                  model1_name : str='model/classification_cnn_model_vis5.h5',
                  model2_name : str='model/status_cnn_model_vis4.h5') -> Dict[str,Any]:
    '''
    Parameters
    ----------
    channels : List[str]
        Namen der übergebenen Kanäle
    data : ndarray
        EEG-Signale der angegebenen Kanäle
    fs : float
        Sampling-Frequenz der Signale.
    reference_system :  str
        Welches Referenzsystem wurde benutzt, "Bezugselektrode", nicht garantiert korrekt!
    model_name : str
        Name eures Models,das ihr beispielsweise bei Abgabe genannt habt. 
        Kann verwendet werden um korrektes Model aus Ordner zu laden
    Returns
    -------
    prediction : Dict[str,Any]
        enthält Vorhersage, ob Anfall vorhanden und wenn ja wo (Onset+Offset)
    '''

#------------------------------------------------------------------------------
# Euer Code ab hier  

    # Initialisiere Return (Ergebnisse)
    seizure_present = True # gibt an ob ein Anfall vorliegt
    seizure_confidence = 0.5 # gibt die Unsicherheit des Modells an (optional)
    onset = 50   # gibt den Beginn des Anfalls an (in Sekunden)
    onset_confidence = 0.50 # gibt die Unsicherheit bezüglich des Beginns an (optional)
    offset = 100 # gibt das Ende des Anfalls an (optional)
    offset_confidence = 0.50   # gibt die Unsicherheit bezüglich des Endes an (optional)

################################################################################################################################################
    filtered_data = np.empty_like(data)

    for i in range(data.shape[0]):
        signal = data[i, :]

        # high-low pass filter
        filtered_signal = mne.filter.filter_data(signal, fs, l_freq=1.0, h_freq=70.0, n_jobs=2, verbose=False)

        # notch filter
        filtered_signal = mne.filter.notch_filter(filtered_signal, fs, freqs=50.0, n_jobs=2, verbose=False)

        # save filtered data
        filtered_data[i, :] = filtered_signal
    data = filtered_data

    desired_channel_count = 19  # 期望的通道数
    current_channel_count = len(data)

    if current_channel_count < desired_channel_count:
        signal = np.empty((19, len(data[0])))
        # the count of channels need to add
        channels_to_add = desired_channel_count - current_channel_count
        # copy channel[0]
        for j in range(channels_to_add):
            signal[i] = data[0]
        data = signal
        
    count_feature_vector1 = 5
    data = np.array(data)
    CSP = np.empty((count_feature_vector1*4, 1))
    # print(W1_name)
    W1 = np.load(W1_name)
    #     Project the source signal onto CSP space
    Zi = np.dot(W1,data) # formular:Zi = W*Xi
    #     Use the logarithm of the variance of the projected signal as a feature
    var_Zi = np.var(Zi,1)

    for f in range(len(var_Zi)):
        CSP[f, :] = np.log(var_Zi[f])
    CSP = CSP.reshape((1, 20, 1))
    
    model1 = load_model(model1_name)

    # prediction
    print('---start prediction---')
    predictions = model1.predict(CSP, verbose=0)
    predictions = float(predictions)
    
    if predictions >= 0.5:
        confidence = 2 * (predictions - 0.5)
        seizure_confidence = confidence
        
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prediction1
        window_size1 = 10  # size of silde window
        stride1 = 10        # stride
        window_samples1 = window_size1 * fs

        predict_result1 = []
        for i in range(0, data.shape[1] - window_samples1 + 1, stride1 * fs):
            # obtain segmentation
            window_slice = data[:, i:i+window_samples1]
            CSP = np.empty((count_feature_vector1*4, 1))

            #     Project the source signal onto CSP space
            Zi = np.dot(W1,window_slice) # formular:Zi = W*Xi
            #     Use the logarithm of the variance of the projected signal as a feature
            var_Zi = np.var(Zi,1)

            for f in range(len(var_Zi)):
                CSP[f, :] = np.log(var_Zi[f])
            CSP = CSP.reshape((1, 20, 1))

            # predict
            prediction = model1.predict(CSP, verbose=0)
            predictions = float(predictions)

            if prediction >= 0.4:
                prediction = 1
                for x in range(window_size1):
                    predict_result1.append(prediction)
            else:
                prediction = 0
                for x in range(window_size1):
                    predict_result1.append(prediction)

        predict_result1 = np.array(predict_result1)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prediction2
        count_feature_vector2 = 4
        model2 = load_model(model2_name)
        W2 = np.load(W2_name)

        window_size2 = 10  # size of silde window
        stride2 = 10        # stride
        window_samples2 = window_size2 * fs
        predict_result2 = []

        for i in range(0, data.shape[1] - window_samples2 + 1, stride2 * fs):
            # obtain segmentation
            window_slice = data[:, i:i+window_samples2]
            CSP = np.empty((count_feature_vector2*4, 1))

            #     Project the source signal onto CSP space
            Zi = np.dot(W2,window_slice) # formular:Zi = W*Xi
            #     Use the logarithm of the variance of the projected signal as a feature
            var_Zi = np.var(Zi,1)

            for f in range(len(var_Zi)):
                CSP[f, :] = np.log(var_Zi[f])
            CSP = CSP.reshape((1, 16, 1))

            # predict
            prediction = model2.predict(CSP, verbose=0)
            predictions = float(predictions)

            if prediction >= 0.35:
                prediction = 1
                for m in range(window_size2):
                    predict_result2.append(prediction)
            else:
                prediction = 0
                for n in range(window_size2):
                    predict_result2.append(prediction)

        predict_result2 = np.array(predict_result2)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prediction3
        window_size3 = 1  # size of silde window
        stride3 = 1       # stride
        window_samples3 = window_size3 * fs
        predict_result3 = []

        for i in range(0, data.shape[1] - window_samples3 + 1, stride3 * fs):
            # obtain segmentation
            window_slice = data[:, i:i+window_samples3]
            CSP = np.empty((count_feature_vector1*4, 1))

            #     Project the source signal onto CSP space
            Zi = np.dot(W1,window_slice) # formular:Zi = W*Xi
            #     Use the logarithm of the variance of the projected signal as a feature
            var_Zi = np.var(Zi,1)

            for f in range(len(var_Zi)):
                CSP[f, :] = np.log(var_Zi[f])
            CSP = CSP.reshape((1, 20, 1))

            # predict
            prediction = model1.predict(CSP, verbose=0)
            predictions = float(predictions)

            if prediction >= 0.4:
                prediction = 1
                predict_result3.append(prediction)
            else:
                prediction = 0
                predict_result3.append(prediction)

        predict_result3 = np.array(predict_result3)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~prediction4
        window_size4 = 1  # size of silde window
        stride4 = 1       # stride
        window_samples4 = window_size4 * fs
        predict_result4 = []

        for i in range(0, data.shape[1] - window_samples4 + 1, stride4 * fs):
            # obtain segmentation
            window_slice = data[:, i:i+window_samples4]
            CSP = np.empty((count_feature_vector2*4, 1))

            #     Project the source signal onto CSP space
            Zi = np.dot(W2,window_slice) # formular:Zi = W*Xi
            #     Use the logarithm of the variance of the projected signal as a feature
            var_Zi = np.var(Zi,1)

            for f in range(len(var_Zi)):
                CSP[f, :] = np.log(var_Zi[f])
            CSP = CSP.reshape((1, 16, 1))

            # predict
            prediction = model2.predict(CSP, verbose=0)
            predictions = float(predictions)

            if prediction >= 0.35:
                prediction = 1
                predict_result4.append(prediction)
            else:
                prediction = 0
                predict_result4.append(prediction)
                
        
        predict_result4 = np.array(predict_result4)
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~        
        predict_result = predict_result1 * predict_result2 * predict_result3[:len(predict_result1)] * predict_result4[:len(predict_result1)]
        predict_result[0] = 0
        predict_result[-1] = 0
        final_result = np.copy(predict_result)

        # find the first '1' and last '1' as onset/offset
        for i in range(1, len(predict_result) - 1):
            if predict_result[i] == 1 and predict_result[i - 1] == 0 and predict_result[i + 1] == 0:
                final_result[i] = 0
        # print(final_result.shape)

        indices = np.where(final_result == 1)[0]
        if len(indices) > 3:
            onset = indices[0] + 1
            offset = indices[-1] + 1
            seizure_present = True
            print('seizure')
            print('seizure confidence:', seizure_confidence)
            print('onset:', onset)
            print('offset:', offset)

        else: # if seizure time less than 3s
            onset = 0.0
            offset = 0.0
            seizure_present = False
            print('no seizure')
            print('no seizure confidence:', seizure_confidence)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    else:
        seizure_present = False
        confidence = 2 * (0.5 - predictions)
        seizure_confidence = confidence
        onset = 0.0
        offset = 0.0
        print('no seizure')
        print('no seizure confidence:', seizure_confidence)
    print('---prediction done---')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
################################################################################################################################################

#------------------------------------------------------------------------------  
    prediction = {"seizure_present":seizure_present,"seizure_confidence":seizure_confidence,
                   "onset":onset,"onset_confidence":onset_confidence,"offset":offset,
                   "offset_confidence":offset_confidence}
  
    return prediction # Dictionary mit prediction - Muss unverändert bleiben!
                               
                               
        
