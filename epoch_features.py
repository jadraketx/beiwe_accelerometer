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

def get_calibration_ceof(calibration_dir, id):
        
        if os.path.exists(calibration_dir) == False:
            logging.error(f"Recalibration requires output from calibrate.py. Expected directory {calibration_dir}")
            sys.exit(1)
        coefFile = os.path.join(calibration_dir, id+"_cal_coef.json")
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

def get_acc_stats(epochs):

    t1 = time.perf_counter()

    n = len(epochs)
    if n == 0:
        logging.error("No epoch data")
        sys.exit(1)
    logging.info(f"Computing statistics for {n} epochs")

    stats = [0] * len(epochs)
    #stats = list()
    i = 0
    for t, df in epochs:
        nsamples = len(df)
        if nsamples > 2:
            x = df[X_COL]
            y = df[Y_COL]
            z = df[Z_COL]

            xMean = x.mean()
            yMean = y.mean()
            zMean = z.mean()
            xMin = x.min()
            yMin = y.min()
            zMin = z.min()
            xMax = x.max()
            yMax = y.max()
            zMax = z.max()
            xRange = xMax - xMin
            yRange = yMax - yMin
            zRange = zMax - zMin
            xStd = x.std()
            yStd = y.std()
            zStd = z.std()
            
            vm = (x**2 + y**2 + z**2)**(1/2)
            enmo = vm - 1
            enmo[enmo < 0] = 0
            enmoMean = enmo.mean()
            enmoStd = enmo.std()
            enmoMin = enmo.min()
            enmoMax = enmo.max()
            enmoRange = enmoMax - enmoMin

            stats[i] = ({
                "time":t, "nsamples":nsamples,
                "xMean":xMean, "yMean":yMean, "zMean":zMean,
                "xMin":xMin, "yMin":yMin, "zMin":zMin,
                "xMax":xMax, "yMax":yMax, "zMax":zMax,
                "xRange":xRange, "yRange":yRange, "zRange":zRange,
                "xStd":xStd, "yStd":yStd, "zStd":zStd,
                "enmoMean":enmoMean, "enmoStd":enmoStd, "enmoMin":enmoMin, 
                "enmoMax":enmoMax, "enmoRange":enmoRange
            })
        else:
            stats[i] = ({
                "time":t, "nsamples":nsamples,
                "xMean":pd.NA, "yMean":pd.NA, "zMean":pd.NA,
                "xMin":pd.NA, "yMin":pd.NA, "zMin":pd.NA,
                "xMax":pd.NA, "yMax":pd.NA, "zMax":pd.NA,
                "xRange":pd.NA, "yRange":pd.NA, "zRange":pd.NA,
                "xStd":pd.NA, "yStd":pd.NA, "zStd":pd.NA,
                "enmoMean":pd.NA, "enmoStd":pd.NA, "enmoMin":pd.NA, 
                "enmoMax":pd.NA, "enmoRange":pd.NA
            })
            
        i = i + 1

        if i % 2000 == 0:
            logging.info("{0:2.2f} percent epochs processed".format(i*100/n))

    t2 = time.perf_counter()
    logging.info("Finshed computing epoch statistis {0:8.2f}".format(t2-t1))

    stats = pd.DataFrame(stats)
    stats = stats.set_index("time")
    return(stats)

def compute_epoch_features(params):

    logging.info("-------------EPOCH FEATURES-------------")

    wd = params.workdir
    raw_data_dir = params.datadir
    id = params.id
    epoch_size = params.epoch_size
    scale_g = params.scale_g
    recalibrate = params.recalibrate

    #wd = params['workdir']
    #raw_data_dir = params['datadir']
    #id = params['id']
    #epoch_size = params['epoch_size']
    #scale_g = params['scale_g']
    #recalibrate = params['recalibrate']

    output_dir = os.path.join(wd, "processed_accel")

    #raw data is expected to be organized in subdirectories labeled by user id
    inPath = os.path.join(raw_data_dir, id, "accelerometer")
    if os.path.exists(inPath) == False:
        logging.error(f"Directory does not exist: {inPath}")
    
    #load raw data
    data = load_all_accelerometer_data(inPath, scale_by_g=scale_g)

    #recalibrate: coefficient file assumed to be in [wd]/processed_accel/calibrate_1/[id]/[id]_calib_coef.json
    if recalibrate:
        calibration_summary = {}

        calibration_dir = os.path.join(output_dir, "calibration_1", id)
        intercept, slope = get_calibration_ceof(calibration_dir, id)
        
        data[X_COL] = data[X_COL] * slope['xSlope'] + intercept['xIntercept']
        data[Y_COL] = data[Y_COL] * slope['ySlope'] + intercept['yIntercept']
        data[Z_COL] = data[Z_COL] * slope['zSlope'] + intercept['zIntercept']

    #group into minute epochs
    epochs = data.resample(str(epoch_size)+"s")
    epoch_stats = get_acc_stats(epochs)
    epoch_stats = find_non_wear_segments(epoch_stats, sd_non_wear_threshold=STATIONARY_CUTOFF, non_wear_window=NON_WEAR_WINDOW)

    epoch_dir = os.path.join(output_dir, "epoch_features_2")
    os.makedirs(epoch_dir, exist_ok=True)
    user_out = os.path.join(epoch_dir, id)
    os.makedirs(user_out, exist_ok=True)

    logging.info(f"Output directory set to {user_out}")
    outFile = os.path.join(user_out, id+"_epoch_features.csv")
    logging.info(f"Writing epoch features data to {outFile}")
    #save epochs that have great than 2 observations
    epoch_stats = epoch_stats[epoch_stats['nsamples'] > 2]
    epoch_stats.to_csv(outFile)

    logging.info("--------EPOCH FEATURE PROCESSING COMPLETE--------")
    


if __name__ == "__main__":

    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True, type=str)
    parser.add_argument('--datadir', required=True, type=str)
    parser.add_argument('--id', required=True,type=str)
    parser.add_argument('--recalibrate', action=argparse.BooleanOptionalAction)
    parser.add_argument('--scale-g', action=argparse.BooleanOptionalAction)
    parser.add_argument('--epoch-size', type=int)
    parser.set_defaults(epoch_size=EPOCH_SIZE, rescale=False, scale_g=False)

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
    