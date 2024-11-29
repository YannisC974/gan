# Labeled Retinal Optical Coherence Tomography Dataset for Classification of
# Normal, Drusen, and CNV Cases

# Contributor(s): Saman Sotoudeh-Paima, Fedra Hajizadeh, Hamid Soltanian-Zadeh

# There are two different options for reading the dataset.

# Option 1: Reading all images. This would result in 16822 images.

# Option 2: Reading the worst-case condition images for each volume 
# (i.e., if a patient was detected as a CNV case, only CNV-appearing 
# B-scans were included for training procedure and normal and drusen 
# B-scans of that patient are excluded from the dataset). This would
# result in 12649 images.


# Importing essential libraries

import cv2
import numpy as np
import pandas as pd

csv_path   = '/Users/yannischappetjuan/Desktop/IA/PROJET/data_information.csv'
file_path  = '/Users/yannischappetjuan/Desktop/IA/PROJET/NEH_UT_2021RetinalOCTDataset'

imageSize  = 224

##############################################################################

# Code for option 1
    
def read_all_images(file_path, csv_path):
    
    df = pd.read_csv(csv_path)
    
    X_patient = []
    y_patient = []
    
    for patient_class in np.unique(df['Class']):
        
        df_classwise = df[df['Class'] == patient_class]
        
        for patient_index in np.unique(df_classwise['Patient ID']):
            
            X = []
            y = []
    
            df_patientwise = df_classwise[df_classwise['Patient ID'] == patient_index]
            
            for i in range(len(df_patientwise)):
                
                img = cv2.imread(file_path + df_patientwise.iloc[i]['Directory'])
                img = cv2.resize(img, (imageSize, imageSize))
                
                img = np.asarray(img)
                
                X.append(img)
            
                if df_patientwise.iloc[i]['Label'].lower() == 'normal':
                    
                    y.append(0)
                    
                elif df_patientwise.iloc[i]['Label'].lower() == 'drusen':
                    
                    y.append(1)
                    
                elif df_patientwise.iloc[i]['Label'].lower() == 'cnv':
                    
                    y.append(2)
                    
                print(patient_class, patient_index)
                    
            X_patient.append(X)
            y_patient.append(y)

    return X_patient, y_patient

##############################################################################

# Code for option 2

def read_worstcase_images(file_path, csv_path):
    
    df = pd.read_csv(csv_path)
    
    X_patient = []
    y_patient = []
    
    for patient_class in np.unique(df['Class']):
        
        df_classwise = df[df['Class'] == patient_class]
        
        for patient_index in np.unique(df_classwise['Patient ID']):
            
            X = []
            y = []
    
            df_patientwise = df_classwise[df_classwise['Patient ID'] == patient_index]
            
            for i in range(len(df_patientwise)):
                
                if df_patientwise.iloc[i]['Class'] == df_patientwise.iloc[i]['Label']:
                    
                    img = cv2.imread(file_path + df_patientwise.iloc[i]['Directory'])
                    img = cv2.resize(img, (imageSize, imageSize))
                    
                    img = np.asarray(img)
                    
                    X.append(img)
                
                    if df_patientwise.iloc[i]['Label'].lower() == 'normal':
                        
                        y.append(0)
                        
                    elif df_patientwise.iloc[i]['Label'].lower() == 'drusen':
                        
                        y.append(1)
                        
                    elif df_patientwise.iloc[i]['Label'].lower() == 'cnv':
                        
                        y.append(2)
                        
                    print(patient_class, patient_index)
                        
            X_patient.append(X)
            y_patient.append(y)

    return X_patient, y_patient