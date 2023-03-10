"""
 * @author Monsij Biswal, Nima Namazi
 * @email mbiswal@ucsb.edu
 * @desc Utility functions for ECG dataset
 """

import os
import glob
import pandas as pd
import numpy as np
import subprocess
import geomstats.geometry.spd_matrices as spd
import geomstats.backend as gs
import scipy.io
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

manifold = spd.SPDMatrices(12)

WFDB_dict = {"426177001":"SB Sinus Bradycardia ", #  <--
            "426783006":"SR	Sinus Rhythm", #  <--
            "427084000":"ST	Sinus Tachycardia", #  <--
            "164889003":"AFIB Atrial Fibrillation", # <--
            "426761007":"SVT Supraventricular Tachycardia", 
            "713422000":"AT Atrial Tachycardia",
            "164890007":"AF	Atrial Flutter",
            "251146004":"LVQRSAL lower voltage QRS in all lead",
            "17366009":"Not added",
            "164865005":"Not added",
            "233897008":"AVRT Atrioventricular Reentrant Tachycardia",
            "251166008":"Not added",
            "164934002":"TWC T wave Change",
            "164931005":"STTU ST tilt up"
            }

#----------------- Functions for new dataset - prefixed by "dn" -----------------------#

dn_base_path = './ECGDataDenoisedMat/'
diagnostics_path = './Diagnostics.xlsx'
labels_dict = {}    # {patient_id : rhythm_acr}
exception_patient_ids = ['MUSE_20181222_204118_08000', 'MUSE_20181222_204121_42000', 
                         'MUSE_20181222_204122_52000', 'MUSE_20181222_204123_64000', 
                         'MUSE_20181222_204128_13000', 'MUSE_20181222_204131_50000',
                         'MUSE_20181222_204140_77000', 'MUSE_20181222_204141_91000', 
                         'MUSE_20181222_204143_03000', 'MUSE_20181222_204146_34000', 
                         'MUSE_20181222_204154_20000', 'MUSE_20181222_204155_31000', 
                         'MUSE_20181222_204156_45000', 'MUSE_20181222_204157_58000', 
                         'MUSE_20181222_204158_72000', 'MUSE_20181222_204207_92000', 
                         'MUSE_20181222_204212_44000', 'MUSE_20181222_204217_03000', 
                         'MUSE_20181222_204218_14000', 'MUSE_20181222_204219_27000', 
                         'MUSE_20181222_204222_63000', 'MUSE_20181222_204226_00000', 
                         'MUSE_20181222_204227_13000', 'MUSE_20181222_204236_34000', 
                         'MUSE_20181222_204237_47000', 'MUSE_20181222_204239_70000', 
                         'MUSE_20181222_204240_84000', 'MUSE_20181222_204243_08000', 
                         'MUSE_20181222_204245_36000', 'MUSE_20181222_204246_47000', 
                         'MUSE_20181222_204248_77000', 'MUSE_20181222_204249_88000', 
                         'MUSE_20181222_204302_49000', 'MUSE_20181222_204303_61000', 
                         'MUSE_20181222_204306_99000', 'MUSE_20181222_204309_22000', 
                         'MUSE_20181222_204310_31000', 'MUSE_20181222_204312_58000', 
                         'MUSE_20181222_204314_78000', 'MUSE_20181222_204132_64000']


def get_all_file_paths():
    """
    Returns list of all patient_ids : MUSE_{}_{}_{}
    """
    patient_ids = []
    mat_files = glob.glob(dn_base_path + '*.mat')
    for path in mat_files:
        patient_ids.append(os.path.split(path)[-1][:-4])
    usable_patient_ids = list(set(patient_ids) - set(exception_patient_ids))
    return usable_patient_ids

def init_labels():
    """
    Creates global dictionary for labels of each patient. Call once
    changes the globally defined labels_dict
    """
    diag_xlsx = pd.read_excel(diagnostics_path)
    global labels_dict
    labels_dict = dict(zip(diag_xlsx["FileName"], diag_xlsx["Rhythm"]))

def get_rhythm_acr(patient_id : str):
    """
    Returns rhythm acr for a given patient id
    """
    return labels_dict[patient_id]

def get_random_patient_id():
    """
    Returns a random patient_id from the basepath
    """
    csv_files = glob.glob(dn_base_path + '*.mat')
    idx = np.random.randint(0, len(csv_files))
    patient_id = os.path.split(csv_files[idx])[-1][:-4]
    return patient_id

def check_on_manifold(cov_mat : np.array):
    """
    Checks if the given patient data is on the SPD manifold
    """
    return(gs.all(manifold.belongs(cov_mat)))

def get_single_data(patient_id : str = None):
    """
    Returns ECG data (12 x 5000) array for a patient

    If no patient_id is given, it returns for a  random patient in list
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data = scipy.io.loadmat(dn_base_path+patient_id+'.mat')['val']
    #rhythm_acr = dn_get_rhythm_acr(patient_id=patient_id)
    
    ecg_data = ecg_data.T
    return ecg_data

def plot_ecg(patient_id: str = None, lead:int = -1):
    """
    Plots time series ECG data (5000 samples) at given lead and patient id

    If no patient_id is given, it plots for a random patient in list
    
    lead : 0-based indexing, by default plots all leads
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    data = get_single_data(patient_id=patient_id)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    if lead==-1:
        for lead in range(12):
            plt.plot(data[lead,:])
    else:
        plt.plot(data[lead, :])
    plt.xlim([-5, 5005])
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('microV')
    plt.show()

def compute_corr_mat(patient_id : str, plot_corr : bool = False):
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data = get_single_data(patient_id)
    curr_std = np.std(ecg_data, axis=1)
    if np.any(curr_std==0):
        print(patient_id)
    corr_mat = np.corrcoef(ecg_data)
    if np.any(curr_std==0):
        print("Zero std found for : ", patient_id)
    corr_mat = np.corrcoef(ecg_data)

    if plot_corr:
        plt.imshow(corr_mat, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()

    return corr_mat

def load_Chapman_ECG(balanced : bool = False):
    """

    """
    
    print("Loading denoised dataset of Chapman Shaoxing 12-lead ECG Data...")
    usable_patient_ids = get_all_file_paths()
    init_labels()
    #usable_patient_ids = list(set(all_patient_ids) - set(exception_patient_ids))
    per_class_count  = np.zeros(4)     # target = 1564
    class_ids = {"SB": 0, "ST": 1, "SR": 2, "AFIB": 3} 
    mat, patient_ids, target = [], [], []
    for patient_id in tqdm(usable_patient_ids):
        tmp = np.zeros((12,12))
        curr_rhythm_acr = get_rhythm_acr(patient_id=patient_id)
        if curr_rhythm_acr in ["AFIB", "SB", "SR", "ST"]:
            if per_class_count[class_ids[curr_rhythm_acr]]==1564 and balanced:
                continue    
            tmp = compute_corr_mat(patient_id=patient_id, plot_corr=False)
            if manifold.belongs(tmp):    # add to list if tmp lies on spd manifold
                mat.append(tmp)
                target.append(curr_rhythm_acr)
                patient_ids.append(patient_id)
                if balanced:
                    per_class_count[class_ids[curr_rhythm_acr]] += 1

    return mat, patient_ids, target

def get_per_class_count(y_test: list):
    """
    Returns dictionary of instance count in y_test
    """
    per_class_count = {'SB':0,
                   'SR':0,
                   'ST':0,
                   'AFIB':0}

    for true_val in y_test:
        per_class_count[true_val] += 1
    return per_class_count

def get_confusion_matrix(y_test : list, y_pred : list, target_class_list : list):
    """
    Returns pandas.DataFrame representing confusion matrix corresponding to y_test and y_pred
    """
    per_class_count = get_per_class_count(y_test=y_test)
    cmat = confusion_matrix(y_test, y_pred, labels=target_class_list).astype('float')  # more data for SB
    pos = 0 
    for i in target_class_list:
        cmat[pos,:] = cmat[pos,:] / per_class_count[i]
        pos += 1
    cmat = np.round(cmat,3)
    index_l = ['true:' + class_name for class_name in target_class_list]
    column_l = ['pred:' + class_name for class_name in target_class_list]
    cmtx = pd.DataFrame(cmat, 
    index=index_l, 
    columns=column_l)
    return cmtx