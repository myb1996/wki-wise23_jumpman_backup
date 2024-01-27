# -*- coding: utf-8 -*-
"""
Diese Datei sollte nicht verändert werden und wird von uns gestellt und zurückgesetzt.

Skript testet das vortrainierte Modell


@author: Maurice Rohr
"""


from predict import predict_labels
from wettbewerb import EEGDataset, save_predictions
import argparse
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict given Model')
    parser.add_argument('--test_dir', action='store',type=str,default='/shared_data/training_mini')
    parser.add_argument('--W1_name', action='store',type=str,default='model/W_vis5.npy')
    parser.add_argument('--W2_name', action='store',type=str,default='model/W_vis4.npy')
    parser.add_argument('--model1_name', action='store',type=str,default='model/classification_cnn_model_vis5.h5')
    parser.add_argument('--model2_name', action='store',type=str,default='model/status_cnn_model_vis4.h5')
    parser.add_argument('--allow_fail',action='store_true',default=False)
    args = parser.parse_args()
    
    # Erstelle EEG Datensatz aus Ordner
    dataset = EEGDataset(args.test_dir)
    print(f"Teste Modell auf {len(dataset)} Aufnahmen")
    
    predictions = list()
    start_time = time.time()
    
    # Rufe Predict Methode für jedes Element (Aufnahme) aus dem Datensatz auf
    for item in dataset:
        id,channels,data,fs,ref_system,eeg_label = item
        try:
            _prediction = predict_labels(channels, data, fs, ref_system, 
                                         W1_name=args.W1_name, W2_name=args.W2_name, 
                                         model1_name=args.model1_name, model2_name=args.model2_name)
            _prediction["id"] = id
            predictions.append(_prediction)
        except:
            if args.allow_fail:
                raise
        
    pred_time = time.time()-start_time
    
    save_predictions(predictions) # speichert Prädiktion in CSV Datei
    print("Runtime",pred_time,"s")
