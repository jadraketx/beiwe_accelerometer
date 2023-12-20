import argparse
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
import json
from utils import load_all_accelerometer_data, init_logger
from settings import EPOCH_SIZE, X_COL, Y_COL, Z_COL, STATIONARY_CUTOFF, NON_WEAR_WINDOW

def get_calibration_ceof(coefFile):
        
        #if os.path.exists(calibration_dir) == False:
        #   logging.error(f"Recalibration requires output from calibrate.py. Expected directory {calibration_dir}")
        #    sys.exit(1)
        #coefFile = os.path.join(calibration_dir, id+"_cal_coef.json")
        if os.path.exists(coefFile) == False:
            logging.error(f"Calibration coefficient file not found: {coefFile}")
            sys.exit(1)
        #read in json file
        with open(coefFile, 'r') as f:
            calibration_summary = json.load(f)
        
        intercept = {
            'xIntercept': calibration_summary['calibration-xOffset(g)'],
            'yIntercept': calibration_summary['calibration-yOffset(g)'],
            'zIntercept': calibration_summary['calibration-zOffset(g)']
        }

        slope = {
            'xSlope': calibration_summary['calibration-xSlope'],
            'ySlope': calibration_summary['calibration-ySlope'],
            'zSlope': calibration_summary['calibration-zSlope']
        }
        logging.info("Offset and gain coefficients loaded for recalibration")
        logging.info("Gain/Slope")
        logging.info(f"\t xSlope: {slope['xSlope']}")
        logging.info(f"\t ySlope: {slope['ySlope']}")
        logging.info(f"\t zSlope: {slope['zSlope']}")

        logging.info("Offset/Intercept")
        logging.info(f"\t xInt: {intercept['xIntercept']}")
        logging.info(f"\t yInt: {intercept['yIntercept']}")
        logging.info(f"\t zInt: {intercept['zIntercept']}")
        return(intercept, slope)

def calc_xyz_sd_nonwear(epoch_stats, min_period=NON_WEAR_WINDOW, sd_cutoff=STATIONARY_CUTOFF):
    #see https://doi.org/10.1080/02640414.2019.1703301
    #method 1: calc variance of acc_x, acc_y, acc_z for each epoch
    #if variance is < 0.003g for 30 consecutive epochs, label all epochs as non-wear
    logging.info(f"Finding non-wear segments: method=SD_XYZ, period={min_period}")
    t1 = time.perf_counter()
    l = len(epoch_stats)
    if l < min_period:
        print("Number of epochs less than minimum period")
        return None

    colname = "xyz_nonwear_" + str(min_period)
    xyz_nonwear = pd.DataFrame({colname:[0]*l},index=epoch_stats.index)
    xyz_candidates = (epoch_stats['x_std'] < sd_cutoff) & (epoch_stats['y_std'] < sd_cutoff) & (epoch_stats['z_std'] < sd_cutoff)
    stop = l - min_period
    for k in range(0,stop):
        window = xyz_candidates[k:(k+min_period)]
        if window.sum() == min_period:
            xyz_nonwear[colname][window.index] = 1
    t2 = time.perf_counter()
    logging.info("Complete ({0:8.2f}s)".format(t2 - t1))
    return(xyz_nonwear)

def calc_vm_sd_nonwear(epoch_stats, min_period=NON_WEAR_WINDOW, sd_cutoff=STATIONARY_CUTOFF):

    logging.info(f"Finding non-wear segments: method=SD_VM, period={min_period}")
    t1 = time.perf_counter()

    #method 2: same thing as above but with variance of the vector magnitude
    l = len(epoch_stats)
    if l < min_period:
        print("Number of epochs less than minimum period")
        return None

    colname = "vm_nonwear_" + str(min_period)
    vm_nonwear = pd.DataFrame({colname:[0]*l},index=epoch_stats.index)
    vm_candidates = epoch_stats['vm_std'] < sd_cutoff
    stop = l - min_period
    for k in range(0,stop):
        window2 = vm_candidates[k:(k+min_period)]
        if window2.sum() == min_period:
            vm_nonwear[colname][window2.index] = 1
    t2 = time.perf_counter()
    logging.info("Complete ({0:8.2f}s)".format(t2 - t1))
    return(vm_nonwear)

def calc_van_hees_nonwear(df, minutes, min_period=NON_WEAR_WINDOW, step=15, sd_cutoff=STATIONARY_CUTOFF):

    logging.info(f"Finding non-wear segments: method=van Hees, period={min_period}")
    t1 = time.perf_counter()

    #https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0022922
    l = len(minutes)
    if l < min_period:
        print("Number of epochs less than minimum period")
        return None
    colname = "van_hees"+"_"+str(min_period)
    van_hees = pd.DataFrame({colname:[0]*l}, index=minutes)
    t = 0
    winSize = min_period
    while (t+winSize) < len(minutes):
        win = minutes[t:(t+winSize+1)]
        tStart = win[0]
        tStop = win[-1]
        
        temp = df[tStart:tStop]
        nsamples = len(temp)
        #print(f"{t}\t{win[0]}\t{win[-1]}\t{nsamples}")
        if nsamples >= 2:
            temp = temp[['x','y','z']]
            temp_stats = temp.agg(["std","min","max"])
            stdev = temp_stats.loc['std']
            r = temp_stats.loc['max']-temp_stats.loc['min']
            #criteria 1
            if sum(stdev < sd_cutoff) >= 2:
                van_hees[colname][win] = 1
            #critera 2
            if sum(r < 0.05) >= 2:
                van_hees[colname][win] = 1
        t = t+step   
    t2 = time.perf_counter()
    logging.info("Complete ({0:8.2f}s)".format(t2 - t1))
    return(van_hees)

def find_non_wear_segments(epochs, sd_non_wear_threshold, non_wear_window):

    #see https://doi.org/10.1080/02640414.2019.1703301
    #method 1: calc variance of acc_x, acc_y, acc_z for each epoch
    #if variance is < 0.004g for 30 consecutive epochs, label all epochs as non-wear

    #method 2: same thing as above but with variance of the vector magnitude
    logging.info("------Finding non-wear segments--------")
    t1 = time.perf_counter()

    l = len(epochs)
    candidates = [0]*l
    timepoints = [0] * l

    i = 0
    for t, row in epochs.iterrows():

        xStd = row['xStd']
        yStd = row['yStd']
        zStd = row['zStd']

        if sum(pd.isna([xStd, yStd, zStd])) > 0:
            candidates[i] = pd.NA
        else:
            if (xStd < sd_non_wear_threshold) & (yStd < sd_non_wear_threshold) & (zStd < sd_non_wear_threshold):
                candidates[i] = 1
        timepoints[i] = t
        i = i + 1

    candidates = pd.DataFrame({"candidates":candidates}, index=timepoints)
    non_wear = pd.DataFrame({"non_wear": ([0] * len(candidates))}, index=timepoints)

    stop = l - non_wear_window

    for k in range(0, stop):
        window = candidates['candidates'][k:(k+non_wear_window)]
        
        temp_sum = window.sum()
        num_na = pd.isna(window).sum()

        # if temp_sum == non_wear_window, then 30 consecutive minutes identified, set all to non_wear
        if temp_sum == (non_wear_window - num_na):
            non_wear['non_wear'][window.index] = 1

    epochs.insert(len(epochs.columns), "non_wear", non_wear)

    t2 = time.perf_counter()
    logging.info("Finished finding non-wear segments ({0:8.2f}s)".format(t2 - t1))
    return(epochs)

def get_epoch_stats(epochs):
    
    t1 = time.perf_counter()
    n = len(epochs)
    if n == 0:
        logging.error("No epoch data")
        sys.exit(1)
    logging.info(f"Computing statistics for {n} epochs")
    epoch_stats = epochs.agg(['mean','median','skew','min','max','std'])
    epoch_stats.columns = [col[0]+"_"+col[1] for col in epoch_stats.columns.values]
    epoch_stats.insert(0, "nsamples", epochs.size())
    t2 = time.perf_counter()
    logging.info("Finshed computing epoch statistis {0:8.2f}".format(t2-t1))
    return(epoch_stats)


def compute_epoch_features(params):

    logging.info("-------------EPOCH FEATURES-------------")

    wd = params.workdir
    raw_data_dir = params.datadir
    id = params.id
    epoch_size = params.epoch_size
    scale_g = params.scale_g

    recalibrate_path = None
    if params.recalibrate is not None:
        recalibrate_path = params.recalibrate

    if os.path.exists(wd) == False:
        logging.error(f"Working directory does not exist: {wd}")
    else:
        logging.info(f"Working directory set to {wd}")

    #raw data is expected to be organized in subdirectories labeled by user id
    inPath = os.path.join(raw_data_dir, id, "accelerometer")
    if os.path.exists(inPath) == False:
        logging.error(f"Directory does not exist: {inPath}")
    
    #load raw data
    data = load_all_accelerometer_data(inPath, scale_by_g=scale_g)

    #recalibrate: coefficient file assumed to be in [wd]/processed_accel/calibrate_1/[id]/[id]_calib_coef.json
    if recalibrate_path:
        calibration_summary = {}

        #calibration_dir = os.path.join(recalibrate_dir, "calibration_1", id)
        intercept, slope = get_calibration_ceof(recalibrate_path)
        
        data[X_COL] = data[X_COL] * slope['xSlope'] + intercept['xIntercept']
        data[Y_COL] = data[Y_COL] * slope['ySlope'] + intercept['yIntercept']
        data[Z_COL] = data[Z_COL] * slope['zSlope'] + intercept['zIntercept']

    #group into minute epochs and calculate stats
    vm = (data[X_COL]**2 + data[Y_COL]**2 + data[Z_COL]**2)**(1/2)
    data.insert(len(data.columns), "vm", vm)
    epochs = data.resample(str(epoch_size)+"s")
    epoch_stats = get_epoch_stats(epochs)

    #find nonwear segments
    xyz_nonwear_15 = calc_xyz_sd_nonwear(epoch_stats, min_period=15)
    xyz_nonwear_30 = calc_xyz_sd_nonwear(epoch_stats, min_period=30)
    xyz_nonwear_60 = calc_xyz_sd_nonwear(epoch_stats, min_period=60)
    vm_nonwear_15 = calc_vm_sd_nonwear(epoch_stats, min_period=15)
    vm_nonwear_30 = calc_vm_sd_nonwear(epoch_stats, min_period=30)
    vm_nonwear_60 = calc_vm_sd_nonwear(epoch_stats, min_period=60)
    van_hees_30 = calc_van_hees_nonwear(data, minutes=list(epochs.groups), min_period=30)
    van_hees_60 = calc_van_hees_nonwear(data, minutes=list(epochs.groups), min_period=60)

    epoch_stats = pd.concat([epoch_stats,xyz_nonwear_15, xyz_nonwear_30, xyz_nonwear_60,vm_nonwear_15,vm_nonwear_30,vm_nonwear_60,van_hees_30,van_hees_60], axis=1)

    outFile = os.path.join(wd, id+"_epoch_features.csv")
    logging.info(f"Writing epoch features data to {outFile}")
    #save epochs that have great than 2 observations
    epoch_stats = epoch_stats[epoch_stats['nsamples'] > 2]
    epoch_stats.to_csv(outFile, index_label="time")

    logging.info("--------EPOCH FEATURE PROCESSING COMPLETE--------")
    


if __name__ == "__main__":

    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True, type=str)
    parser.add_argument('--datadir', required=True, type=str)
    parser.add_argument('--id', required=True,type=str)
    parser.add_argument('--recalibrate', required=False, type=str)
    parser.add_argument('--scale-g', action=argparse.BooleanOptionalAction)
    parser.add_argument('--epoch-size', type=int)
    parser.set_defaults(epoch_size=EPOCH_SIZE, scale_g=False)

    args = parser.parse_args()

    #args = {
    #    'workdir':'/Users/jadrake/Local_dev/beiwe_msk_analysis',
    #    'datadir':'/Users/jadrake/Local_dev/beiwe_msk_analysis/temp_data',
    #    'id':'g5xnzgjd',
    #    'epoch_size':EPOCH_SIZE,
    #    'recalibrate':True,
    #    'scale_g':True
    #}

    compute_epoch_features(args)
    