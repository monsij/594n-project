"""
 * @author Monsij Biswal
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

base_path = './WFDB/'


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

# classes for geometric ML
class_labels = {"426177001":"SB",
                "426783006":"SR",
                "427084000":"ST",
                "164889003":"AF"
}

# contain 0-row which prevents corrcoef computation
exception_patient_ids = ['JS05212','JS07788','JS01266','JS09412',
                       'JS10480','JS00459','JS10475','JS10477',
                        'JS02113','JS06200','JS04173','JS04957',
                        'JS02826','JS01040','JS05815']



def get_all_file_paths():
    """
    Returns list of all patient_ids
    """
    patient_ids = []
    mat_files = glob.glob(base_path + '*.mat')
    for path in mat_files:
        patient_ids.append(os.path.split(path)[-1][:-4])
    return patient_ids


def get_random_patient_id():
    """
    Returns a random patient_id from the basepath
    """
    mat_files = glob.glob(base_path + '*.mat')
    idx = np.random.randint(0, len(mat_files))
    patient_id = os.path.split(mat_files[idx])[-1][:-4]
    return patient_id

def get_rhythm_id(patient_id : str):
    """
    Returns rhythm id for a given patient id
    """
    hea_file = open(base_path + patient_id + '.hea')
    ids = hea_file.readlines()[15][5:].split(',')
    if len(ids)==1:    # trimming \n depending on number of diagnoses
        rhythm_id = ids[0][:-1]
    else:
        rhythm_id = ids[0]
    return rhythm_id


def get_single_data(patient_id : str = None):
    """
    Returns ECG data (12 x 5000) array for a patient along with rhythm name

    If no patient_id is given, it returns for a  random patient in list
    """
    rhythm_acr = ""
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data = scipy.io.loadmat(base_path+patient_id+'.mat')['val']
    rhythm_name = WFDB_dict[get_rhythm_id(patient_id=patient_id)]
    return ecg_data, rhythm_name

def check_on_manifold(cov_mat : np.array):
    """
    Checks if the given patient data is on the SPD manifold
    """
    return(gs.all(manifold.belongs(cov_mat)))

    #print("Percentage of cov mat on manifold: {:.2f}".format((on_manifold/num_files)*100))


def compute_corr_mat(patient_id: str = None, plot_corr = True):
    """
    Computes (and plots) correlation matrix (12 x 12) array for a patient with given id

    If no patient_id is given, it plots for a random patient in list
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    ecg_data, _ = get_single_data(patient_id)
    curr_std = np.std(ecg_data, axis=1)

    corr_mat = np.corrcoef(ecg_data)

    
    
    if plot_corr:
        plt.imshow(corr_mat, cmap='viridis', vmin=-1, vmax=1)
        plt.colorbar()
        plt.show()

    return corr_mat

def plot_ecg(patient_id: str = None, lead:int = 0):
    """
    Plots time series ECG data (5000 samples) at given lead and patient id

    If no patient_id is given, it plots for a random patient in list
    
    lead : 0-based indexing, by default plots all leads
    """
    if patient_id==None:
        patient_id = get_random_patient_id()
    data, _ = get_single_data(patient_id=patient_id)
    
    fig = plt.gcf()
    fig.set_size_inches(12, 6)

    if lead==0:
        for lead in range(12):
            plt.plot(data[lead,:])
    else:
        plt.plot(data[lead, :])
    plt.xlim([-5, 5005])
    plt.grid()
    plt.xlabel('Samples')
    plt.ylabel('microV')
    plt.show()

def get_patients_with_rhythm_id(rhythm_id: str=None):
    result = []
    patient_ids = get_all_file_paths()
    for patient_id in patient_ids:
        curr_rhythm_id = get_rhythm_id(patient_id=patient_id)
        if curr_rhythm_id == rhythm_id:
            result.append(patient_id)
    return result

def load_Chapman_ECG(num_classes, target_class_list, as_vectors=False,):
    """Load data from Chapman ECG dataset.

    Parameters
    ----------
    as_vectors : bool
        Whether to return raw data as vectors or as symmetric matrices.
        Optional, default: False

    Returns
    -------
    mat : array-like, shape=[9049, {[12, 12], 144}
        Connectomes.
    patient_ids : array-like, shape=[9049,]
        Patient unique identifiers
    target : array-like, shape=[9049,]
        Labels, whether patients belong to one of the four classes. 
        See class_labels
    num_classes : int
        No.of different classes data can belong
    target_class_list : list, len = num_classes
        Acronyms of required target classes
    
        
    class_list only among ["SB", "SR", "ST", "AF"]
    No.of relevant files for top-4 classes = 9049
    """
    if as_vectors:
        print('Not implemented for as_vectors=True')
    try:
        assert(len(target_class_list)==num_classes)
    except:
        print("Length of target_class_list must be equal to {}".format(num_classes))
        return
    print("Loading Chapman Shaoxing 12-lead ECG Data...",
          "\nUnpacking data for {} classes only".format(num_classes))
    all_patient_ids = get_all_file_paths()
    usable_patient_ids = list(set(all_patient_ids) - set(exception_patient_ids)) 
    mat, patient_ids, target = [], [], []
    for patient_id in tqdm(usable_patient_ids):
        tmp = np.zeros((12,12))
        if get_rhythm_id(patient_id=patient_id) in class_labels.keys():
            if class_labels[get_rhythm_id(patient_id=patient_id)] in target_class_list:
                tmp = compute_corr_mat(patient_id=patient_id, plot_corr=False)
                mat.append(tmp)
                target.append(class_labels[get_rhythm_id(patient_id=patient_id)])
                patient_ids.append(patient_id)

    return mat, patient_ids, target

def get_per_class_count(y_test: list):
    """
    Returns dictionary of instance count in y_test
    """
    per_class_count = {'SB':0,
                   'SR':0,
                   'ST':0,
                   'AF':0}

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


#----------------- Functions for new dataset - prefixed by "dn" -----------------------#

dn_base_path = './ECGDataDenoisedMat/'
diagnostics_path = './Diagnostics.xlsx'
labels_dict = {}    # {patient_id : rhythm_acr}


def dn_get_all_file_paths():
    """
    Returns list of all patient_ids : MUSE_{}_{}_{}
    """
    patient_ids = []
    mat_files = glob.glob(dn_base_path + '*.mat')
    for path in mat_files:
        patient_ids.append(os.path.split(path)[-1][:-4])
    return patient_ids

def init_labels():
    """
    Creates global dictionary for labels of each patient. Call once
    changes the globally defined labels_dict
    """
    diag_xlsx = pd.read_excel(diagnostics_path)
    global labels_dict
    labels_dict = dict(zip(diag_xlsx["FileName"], diag_xlsx["Rhythm"]))

def dn_get_rhythm_acr(patient_id : str):
    """
    Returns rhythm acr for a given patient id
    """
    return labels_dict[patient_id]

def dn_get_random_patient_id():
    """
    Returns a random patient_id from the basepath
    """
    csv_files = glob.glob(dn_base_path + '*.mat')
    idx = np.random.randint(0, len(csv_files))
    patient_id = os.path.split(csv_files[idx])[-1][:-4]
    return patient_id

def dn_get_single_data(patient_id : str = None):
    """
    Returns ECG data (12 x 5000) array for a patient

    If no patient_id is given, it returns for a  random patient in list
    """
    if patient_id==None:
        patient_id = dn_get_random_patient_id()
    ecg_data = scipy.io.loadmat(dn_base_path+patient_id+'.mat')['val']
    #rhythm_acr = dn_get_rhythm_acr(patient_id=patient_id)
    
    ecg_data = ecg_data.T
    return ecg_data

def dn_compute_corr_mat(patient_id : str, plot_corr : bool = False):
    if patient_id==None:
        patient_id = dn_get_random_patient_id()
    ecg_data = dn_get_single_data(patient_id)
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



def dn_load_Chapman_ECG():
    """

    """
    
    print("Loading denoised dataset of Chapman Shaoxing 12-lead ECG Data...")
    all_patient_ids = dn_get_all_file_paths()
    init_labels()
    #usable_patient_ids = list(set(all_patient_ids) - set(exception_patient_ids)) 
    mat, patient_ids, target = [], [], []
    for patient_id in tqdm(all_patient_ids):
        tmp = np.zeros((12,12))
        if dn_get_rhythm_acr(patient_id=patient_id) in ["AFIB", "SB", "SR", "ST"]:
                tmp = dn_compute_corr_mat(patient_id=patient_id, plot_corr=False)
                if manifold.belongs(tmp):    # add to list if tmp lies on spd manifold
                    mat.append(tmp)
                    target.append(dn_get_rhythm_acr(patient_id=patient_id))
                    patient_ids.append(patient_id)

    return mat, patient_ids, target


def load_mat(patient_id : str):
    """
    can delete
    Convert .csv ecg datafile to .mat (maybe faster) ?
    """
    ecg_data = scipy.io.loadmat('./ECGDataDenoisedMat/' +patient_id+'.mat')['val']
    print(ecg_data[0,:])

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