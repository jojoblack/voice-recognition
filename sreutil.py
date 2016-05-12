import sys,os
import numpy as np
import random
import math
from gmm import *

vad_time_delete = 50
vad_time_connect = 50
vad_selection_param = 2.0
    
def inner_vad_to_sd_scp(label):
    """
    Brief:
    vad sequences to speaker diarization type
    """
    frame_num = len(label)
    flag = label[0]
    time_beg = 0
    seg = []
    for idx in range(1, frame_num):
        if idx == frame_num - 1:
            time_end = frame_num
            if flag == True:
                seg.append((time_beg, time_end))
        elif label[idx] == flag:
            continue
        else:
            time_end = idx
            #print time_end
            if flag == True:
                seg.append((time_beg, time_end))
            flag = label[idx]
            time_beg = time_end

    sph_segs = []
    time_beg = seg[0][0]
    time_end = seg[0][1]
    for i in range(1, len(seg)):
        var_beg = seg[i][0]
        if var_beg - time_end < vad_time_connect:
            time_end = seg[i][1]
        else:
            if time_end - time_beg > vad_time_delete:
                sph_segs.append([time_beg, time_end])
            time_beg = seg[i][0]
            time_end = seg[i][1]
    if time_end - time_beg > vad_time_delete:
        sph_segs.append([time_beg, time_end])
    return sph_segs

def seg_to_label(segs, n_frame):
    label = np.zeros(n_frame)

    for seg in segs:
        idx_beg = seg[0]
        idx_end = seg[1]
        for i in range(idx_beg,idx_end):
            label[i] = 1
    label = (label==1)
    return label

def vad_post_proc(label):
    n_frame = len(label)
    segs = inner_vad_to_sd_scp(label)
    valid_len = 0
    for seg in segs:
        valid_len += seg[1] - seg[0]

    new_label = seg_to_label(segs, n_frame)
    assert((len(np.nonzero(new_label)[0]) == valid_len) and (len(new_label) == n_frame))
    return new_label


def vad(energy):
    """ 
    Brief:
    voice activity detection
    """
    mean = np.mean(energy)
    cov = np.cov(energy)
    energy = (energy - mean) / np.sqrt(cov)
    gmm = GMM(n_dim = 1, n_mix = 3)
    gmm.train(energy[:, np.newaxis])
    # print energy
    idx = np.nonzero(gmm.means == gmm.means.max())
    thd = gmm.means[idx] - vad_selection_param * np.sqrt(gmm.covs[idx])
    label = energy > thd
    label = vad_post_proc(label)
    return label
