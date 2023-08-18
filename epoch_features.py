import argparse
import logging
import os
import sys
import time
import pandas as pd
import numpy as np
import json
from utils import load_all_accelerometer_data, init_logger
from settings import EPOCH_SIZE, X_COL, Y_COL, Z_COL

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

def compute_epoch_features(params):

    logging.info("-------------EPOCH FEATURES-------------")

    #wd = params.workdir
    #raw_data_dir = params.datadir
    #id = params.id
    #epoch_size = params.epoch_size
    #scale_g = params.scale_g
    #recalibrate = params.recalibrate

    wd = params['workdir']
    raw_data_dir = params['datadir']
    id = params['id']
    epoch_size = params['epoch_size']
    scale_g = params['scale_g']
    recalibrate = params['recalibrate']

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
        

    


if __name__ == "__main__":

    init_logger()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--workdir', required=True, type=str)
    #parser.add_argument('--datadir', required=True, type=str)
    #parser.add_argument('--id', required=True,type=str)
    #parser.add_argument('--recalibrate', action=argparse.BooleanOptionalAction)
    # parser.add_argument('--scale-g', action=argparse.BooleanOptionalAction)
    #parser.add_argument('--epoch-size', type=int)
    #parser.set_defaults(epoch_size=EPOCH_SIZE, rescale=False, scale_g=False)

    #args = parser.parse_args()

    args = {
        'workdir':'/Users/jadrake/Local_dev/beiwe_msk_analysis',
        'datadir':'/Users/jadrake/Local_dev/beiwe_msk_analysis/temp_data',
        'id':'g5xnzgjd',
        'epoch_size':EPOCH_SIZE,
        'recalibrate':True,
        'scale_g':True
    }

    compute_epoch_features(args)
    