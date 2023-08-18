import argparse
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
import json
from utils import load_all_accelerometer_data
import statsmodels.api as sm
import pprint
from settings import STATIONARY_CUTOFF, STATIONARY_EPOCH_SIZE, MAXITER, IMPROV_TOL, ERR_TOL, CALIB_CUBE, CALIB_MIN_SAMPLES

def init_logger(level="DEBUG"):
    numeric_level = getattr(logging, level.upper(), None)
    logging.basicConfig(level=numeric_level)

def find_stationary_epochs(df):
    
    t1 = time.perf_counter()
    logging.info("Identifying stationary epochs")
    #roughly following van Hees https://doi.org/10.1152/japplphysiol.00421.2014
    #identify epochs where std(x), std(y), std(z) < 0.004g
    epochs = df.resample(STATIONARY_EPOCH_SIZE)
    std_df = epochs.std() < STATIONARY_CUTOFF
    inds = std_df.all(axis='columns')
    stationary_epochs = inds.index[inds]
    res = [0]*len(stationary_epochs)
    for i in range(0,len(stationary_epochs)):
        avg = epochs.get_group(stationary_epochs[i]).mean()
        res[i] = {"xMean":avg[0],"yMean":avg[1],"zMean":avg[2]}
    res = pd.DataFrame(res)

    t2 = time.perf_counter()
    logging.info("Found {0:7d} stationary epochs ({1:8.2f})".format(len(res), t2-t1 ))
    return(res)

def storeCalibrationInformation(
        summary, bestIntercept, bestSlope, bestSlopeT,
        initErr, bestErr, nStatic, iterations, calibratedOnOwnData, goodCalibration
):
    """Store calibration information to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) bestIntercept: Best x/y/z intercept values
    :param list(float) bestSlope: Best x/y/z slope values
    :param list(float) bestSlopeT: Best x/y/z temperature slope values
    :param float initErr: Error (in mg) before calibration
    :param float bestErr: Error (in mg) after calibration
    :param int nStatic: number of stationary points used for calibration
    :param calibratedOnOwnData: Whether params were self-derived
    :param goodCalibration: Whether calibration succeded

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # store output to summary dictionary
    storeCalibrationParams(summary, bestIntercept, bestSlope, bestSlopeT)
    summary['calibration-errsBefore(mg)'] = initErr * 1000
    summary['calibration-errsAfter(mg)'] = bestErr * 1000
    summary['calibration-numStaticPoints'] = nStatic
    summary['calibration-iterations'] = iterations
    summary['quality-calibratedOnOwnData'] = calibratedOnOwnData
    summary['quality-goodCalibration'] = goodCalibration


def storeCalibrationParams(summary, xyzOff, xyzSlope, xyzSlopeT):
    """Store calibration parameters to output summary dictionary

    :param dict summary: Output dictionary containing all summary metrics
    :param list(float) xyzOff: intercept [x, y, z]
    :param list(float) xyzSlope: slope [x, y, z]
    :param list(float) xyzSlopeT: temperature slope [x, y, z]

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    # store output to summary dictionary
    summary['calibration-xOffset(g)'] = xyzOff[0]
    summary['calibration-yOffset(g)'] = xyzOff[1]
    summary['calibration-zOffset(g)'] = xyzOff[2]
    summary['calibration-xSlope'] = xyzSlope[0]
    summary['calibration-ySlope'] = xyzSlope[1]
    summary['calibration-zSlope'] = xyzSlope[2]
    summary['calibration-xSlopeTemp'] = xyzSlopeT[0]
    summary['calibration-ySlopeTemp'] = xyzSlopeT[1]
    summary['calibration-zSlopeTemp'] = xyzSlopeT[2]

def getCalibrationCoefs(data, summary):
    """
    Code from biobank: https://github.com/OxWearables/biobankAccelerometerAnalysis/blob/master/src/accelerometer/device.py
    Identify calibration coefficients from java processed file

    Get axes offset/gain/temp calibration coefficients through linear regression
    of stationary episodes
    :param str stationaryFile: Output/temporary file for calibration
    :param dict summary: Output dictionary containing all summary metrics

    :return: Calibration summary values written to dict <summary>
    :rtype: void
    """

    t1 = time.perf_counter()
    logging.info("Computing calibration coefficients")

    xyz = data[['xMean', 'yMean', 'zMean']].to_numpy()
    if 'temp' in data:
        T = data['temp'].to_numpy()
    else:  # use a dummy
        T = np.zeros(len(xyz), dtype=xyz.dtype)

    # Remove any zero vectors as they cause nan issues
    nonzero = np.linalg.norm(xyz, axis=1) > 1e-8
    xyz = xyz[nonzero]
    T = T[nonzero]

    intercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
    slope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
    slopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
    bestIntercept = np.copy(intercept)
    bestSlope = np.copy(slope)
    bestSlopeT = np.copy(slopeT)

    curr = xyz
    target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

    errors = np.linalg.norm(curr - target, axis=1)
    err = np.mean(errors)  # MAE more robust than RMSE. This is different from the paper
    initErr = err
    bestErr = 1e16
    nStatic = len(xyz)


    # Check that we have enough uniformly distributed points:
    # need at least one point outside each face of the cube
    if len(xyz) < CALIB_MIN_SAMPLES or (np.max(xyz, axis=0) < CALIB_CUBE).any() or (np.min(xyz, axis=0) > -CALIB_CUBE).any():
        goodCalibration = 0

    else:  # we do have enough uniformly distributed points

        for it in range(MAXITER):

            # Weighting. Outliers are zeroed out
            # This is different from the paper
            maxerr = np.quantile(errors, .995)
            weights = np.maximum(1 - errors / maxerr, 0)

            # Optimize params for each axis
            for k in range(3):

                inp = curr[:, k]
                out = target[:, k]
                inp = np.column_stack((inp, T))
                inp = sm.add_constant(inp, prepend=True, has_constant='add')
                params = sm.WLS(out, inp, weights=weights).fit().params
                # In the following,
                # intercept == params[0]
                # slope == params[1]
                # slopeT == params[2]
                intercept[k] = params[0] + (intercept[k] * params[1])
                slope[k] = params[1] * slope[k]
                slopeT[k] = params[2] + (slopeT[k] * params[1])

            # Update current solution and target
            curr = intercept + (xyz * slope)
            curr = curr + (T[:, None] * slopeT)
            target = curr / np.linalg.norm(curr, axis=1, keepdims=True)

            # Update errors
            errors = np.linalg.norm(curr - target, axis=1)
            err = np.mean(errors)
            errImprov = (bestErr - err) / bestErr

            if err < bestErr:
                bestIntercept = np.copy(intercept)
                bestSlope = np.copy(slope)
                bestSlopeT = np.copy(slopeT)
                bestErr = err

            if errImprov < IMPROV_TOL:
                break

        goodCalibration = int(not ((bestErr > ERR_TOL) or (it + 1 == MAXITER)))

    if goodCalibration == 0:  # restore calibr params
        bestIntercept = np.array([0.0, 0.0, 0.0], dtype=xyz.dtype)
        bestSlope = np.array([1.0, 1.0, 1.0], dtype=xyz.dtype)
        bestSlopeT = np.array([0.0, 0.0, 0.0], dtype=T.dtype)
        bestErr = initErr
        
    storeCalibrationInformation(
        summary,
        bestIntercept=bestIntercept,
        bestSlope=bestSlope,
        bestSlopeT=bestSlopeT,
        initErr=initErr,
        bestErr=bestErr,
        nStatic=nStatic,
        iterations=it,
        calibratedOnOwnData=1,
        goodCalibration=goodCalibration
    )

    t2 = time.perf_counter()
    logging.info("Finished computing calibration coefficients ({0:8.2f})".format(t2-t1))
    return 

def calibrate(params):
    
    logging.info("-------------CALIBRATION-------------")
    
    wd = raw_data_dir = id = user_os = output_dir = ""
    save_stationary = False
    if "workdir" in params:
        wd = params['workdir']
    if "raw_data_dir" in params:
        raw_data_dir = params['raw_data_dir']
    if "id" in params:
        id = params['id']
    if "os" in params:
        user_os = params['os']
    if "save_stationary" in params:
        if params['save_stationary'] == "true":
            save_stationary = True



    #raw data is expected to be organized in subdirectories labeled by user id
    inPath = os.path.join(raw_data_dir, id, "accelerometer")
    if os.path.exists(inPath) == False:
        logging.error(f"Directory does not exist: {inPath}")
    
    scale_by_g = False
    if user_os == 'android':
        scale_by_g = True
    data = load_all_accelerometer_data(inPath, scale_by_g=scale_by_g)
    stationary_epochs = find_stationary_epochs(data)

    calibration_summary = {}
    getCalibrationCoefs(stationary_epochs, calibration_summary)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(calibration_summary)

    #OUTPUT
    #create output directory for processed data
    output_dir = os.path.join(wd, "processed_accel")
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory set to {output_dir}")

    #save stationary raw data to file
    calibration_dir = os.path.join(output_dir, "calibration_1")
    os.makedirs(calibration_dir, exist_ok=True)

    user_out = os.path.join(calibration_dir, id)
    os.makedirs(user_out, exist_ok=True)

    if save_stationary:
        stationary_epochs.to_csv(os.path.join(user_out, id + "_stationary_epochs.csv"), index=False)
    with open(os.path.join(user_out, id+"_cal_coef.json"), 'w') as f:
        json.dump(calibration_summary, f)


if __name__ == "__main__":

    init_logger()

    #parser = argparse.ArgumentParser()
    #parser.add_argument('--workdir', required=True)
    #parser.add_argument('--id', required=True)
    #parser.add_argument('--os', required=True)
    #parser.add_argument('--outdir', required=True)

    #args = parser.parse_args()
    #workdir = args.workdir
    #id = args.id
    #os = args.os
    #outdir = args.outdir
    params = {
        "workdir":"/Users/jadrake/Local_dev/beiwe_msk_analysis/",
        "raw_data_dir":"/Users/jadrake/Local_dev/beiwe_msk_analysis/temp_data",
        "id":"g5xnzgjd",
        "os":"android",
        "save_stationary":"true"
    }

    
    calibrate(params)