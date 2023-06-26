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
import networkx as nx
import mne_connectivity as conny

# Set environment variable so solve issue with parallel crash
# https://stackoverflow.com/questions/40115043/no-space-left-on-device-error-while-fitting-sklearn-model/49154587#49154587
os.environ["JOBLIB_TEMP_FOLDER"] = "/tmp"

# Define paths
path_in = "/mnt/data/arnauOn8TB/vs_resting_state/2_source_space/"
path_out = "/mnt/data/arnauOn8TB/vs_resting_state/3_connectivity_data/"

# Define datasets
datasets = glob.glob(f"{path_in}/*_source_data.joblib")

# Connectivity function
def get_connectivity(todo, n_signals, con_methods):

    # Load data
    data = joblib.load(todo["dataset"])

    # Compile metadata
    output = {
        "id": data["id"],
        "labels": data["labels"],
        "freqbands": ["delta", "theta", "alpha_lo", "alpha_hi", "beta"],
        "eyes": todo["eyes"],
        "session": todo["session"],
    }

    # Reduce epochs
    data = data["data"][todo["epoch_idx"], :, :].copy()

    # Create a network graph
    G = nx.complete_graph(n_signals)

    # Generate a list of edges
    edge_list = ([], [])
    for edge in G.edges():
        edge_list[0].append(edge[0])
        edge_list[1].append(edge[1])

    # Define frequencies
    # TODO: Adjust frequencies to IAF (as a second analysis)
    freqs_to_use = np.arange(2, 31)

    # Calculate connectivities and average across epochs
    con = conny.spectral_connectivity_time(
        data,
        freqs=freqs_to_use,
        method=con_methods,
        indices=edge_list,
        mode="multitaper",
        sfreq=200,
        n_cycles=5,
        fmin=(2, 4, 8, 11, 16),
        fmax=(3, 7, 10, 13, 30),
        faverage=True,
        n_jobs=1,
    )

    # Average across trials
    con = [x.combine(combine="mean") for x in con]

    # Append conny to output
    output["wpli"] = con[0]
    output["coh"] = con[1]

    # Specify out file name
    out_file = os.path.join(
        path_out,
        f"connydat_{output['id']}_eyes_{todo['eyes']}_session_{todo['session']}.joblib",
    )

    # Save
    joblib.dump(output, out_file)

    return con


# Loop!
todo_list = []
for dataset in datasets:

    # Load dataset
    data = joblib.load(dataset)

    # Create todo-list of dataset by condition dictionaries
    for eyes_idx in [0, 1]:
        for session_idx in [1, 2]:
            epoch_idx = (data["trialinfo"][:, 0] == eyes_idx) & (
                data["trialinfo"][:, 1] == session_idx
            )
            todo_list.append(
                {
                    "dataset": dataset,
                    "eyes": "open" if eyes_idx == 1 else "closed",
                    "session": session_idx,
                    "epoch_idx": epoch_idx,
                }
            )

# get dims
n_epochs, n_signals, n_time = data["data"].shape

# Define connectivity methods
con_methods = ["wpli", "coh"]

# Parallel calculation of connectivities
out = joblib.Parallel(n_jobs=-1)(
    joblib.delayed(get_connectivity)(todo, n_signals, con_methods) for todo in todo_list
)
