#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 2023

@author: Stefan Arnau
"""

# Imports
import glob
import os
import joblib
import numpy as np
import scipy.signal
import scipy.io
import mne

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data/arnauOn8TB/vs_resting_state/1_cleaned/"
path_out = "/mnt/data/arnauOn8TB/vs_resting_state/2_source_space/"

# Define datasets
datasets = glob.glob(f"{path_in}/*_cleaned.set")

# Create a template forward solution
trans = "fsaverage"

# Fetch template data
fs_dir = mne.datasets.fetch_fsaverage()
subjects_dir = os.path.dirname(fs_dir)

# Define template source space
src = os.path.join(fs_dir, "bem", "fsaverage-ico-5-src.fif")

# Define template bem
bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

# Load a dataset
epochs = mne.io.read_epochs_eeglab(datasets[0]).apply_baseline(baseline=(-0.2, 0))

# Set standard montage
montage = mne.channels.make_standard_montage("standard_1005")
epochs.set_montage(montage)
epochs.set_eeg_reference(projection=True)

# Setup source space and compute forward solution
fwd = mne.make_forward_solution(
    epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=-2
)

# Get labels for FreeSurfer 'aparc' cortical parcellation with 34 labels/hemi
parcellation_to_use = "PALS_B12_Brodmann"
labels = mne.read_labels_from_annot(
    "fsaverage", parc=parcellation_to_use, subjects_dir=subjects_dir,hemi="both",
)      
          
# Get the ones with names starting with "B" (for Brodmann)
labels = [label for label in labels if label.name[0] == "B"]

# Get label names
label_names = [label.name for label in labels]

# Plot that
Brain = mne.viz.get_brain_class()
brain = Brain(
    "fsaverage",
    "both",
    "inflated",
    subjects_dir=subjects_dir,
    cortex="low_contrast",
    background="white",
    size=(800, 600),
)
brain.add_annotation(parcellation_to_use)

# Loop datasets
for dataset in datasets:

    # Get id string for consistency in file naming
    id_string = dataset.split("/")[-1].split("_")[0]
    
    # Load EEG
    epochs = mne.io.read_epochs_eeglab(dataset).apply_baseline(baseline=(-0.2, 0))

    # Set standard montage
    montage = mne.channels.make_standard_montage("standard_1005")
    epochs.set_montage(montage)
    epochs.set_eeg_reference(projection=True)

    # Load trialinfo
    trialinfo = scipy.io.loadmat(dataset)["trialinfo"]

    # Compute a regularized covariance matrix
    cov = mne.compute_covariance(epochs, tmin=-2.0, tmax=1.995)

    # Compute inverse solution on epoched data
    inverse_operator = mne.minimum_norm.make_inverse_operator(
        epochs.info, fwd, cov, verbose=True
    )

    # Apply inverse solution on epoched data
    snr = 3
    lambda2 = 1.0 / snr**2
    method = "dSPM"
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inverse_operator,
        lambda2,
        method,
        nave=1,
        pick_ori="normal",
        return_generator=True,
    )

    # Average the source estimates within each label
    src = inverse_operator["src"]
    label_ts = mne.extract_label_time_course(
        stcs, labels, src, mode="mean_flip", return_generator=False
    )

    # Compile dict
    source_data = {
        "id": id_string,
        "data": np.stack(label_ts),
        "labels": labels,
        "times": epochs.times,
        "trialinfo": trialinfo,
    }

    # Save
    out_file = os.path.join(path_out, f"{id_string}_source_data.joblib")
    joblib.dump(source_data, out_file)

