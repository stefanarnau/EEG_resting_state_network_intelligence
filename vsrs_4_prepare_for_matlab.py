#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:06:39 2023

@author: plkn
"""
# Imports
import numpy as np
import glob
import joblib
import os
import scipy.io
import networkx as nx

# Paths
path_in = "/mnt/data_dump/vs_resting_state/3_connectivity_data_iaf/"
path_out = "/mnt/data_dump/vs_resting_state/connectivity_data_for_matlab_iaf/"

# get id numbers as strings
id_strings = [
    x.split("connydat")[1].split("_")[2]
    for x in glob.glob(f"{path_in}/*_eyes_open_session_1.joblib")
]

# Loop ids
for id_string in id_strings:

    # Load data
    eo1 = joblib.load(
        os.path.join(path_in, f"connydat_iaf_{id_string}_eyes_open_session_1.joblib")
    )
    eo2 = joblib.load(
        os.path.join(path_in, f"connydat_iaf_{id_string}_eyes_open_session_2.joblib")
    )
    ec1 = joblib.load(
        os.path.join(path_in, f"connydat_iaf_{id_string}_eyes_closed_session_1.joblib")
    )
    ec2 = joblib.load(
        os.path.join(path_in, f"connydat_iaf_{id_string}_eyes_closed_session_2.joblib")
    )

    # Concatenate conditions for both both connectivity measures
    coh = np.stack(
        (eo1["coh"]._data, eo2["coh"]._data, ec1["coh"]._data, ec2["coh"]._data)
    )
    wpli = np.stack(
        (eo1["wpli"]._data, eo2["wpli"]._data, ec1["wpli"]._data, ec2["wpli"]._data)
    )

    # Create np arrays of objects as dimension descriptors
    label_names = np.asarray([label.name for label in eo1["labels"]], dtype="object")
    freqbands = np.asarray(eo1["freqbands"], dtype="object")
    sessions = np.asarray(["open_1", "open_2", "closed_1", "closed_2"], dtype="object")
    dimensions = np.asarray(["session", "edge", "freqband"], dtype="object")

    # Generate a list of edges
    G = nx.complete_graph(82)
    edge_list = ([], [])
    for edge in G.edges():
        edge_list[0].append(edge[0])
        edge_list[1].append(edge[1])
    edge_list = np.stack((np.asarray(edge_list[0]), np.asarray(edge_list[1]))).T + 1

    # Compile dict for subject
    out = {
        "coh": coh,
        "wpli": wpli,
        "label_names": label_names,
        "freqbands": freqbands,
        "sessions": sessions,
        "dimensions": dimensions,
        "edge_list": edge_list,
    }

    # Save
    scipy.io.savemat(
        os.path.join(path_out, f"{id_string}_connectivity_data_iaf.mat"), out
    )
