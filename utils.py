import pandas as pd
import logging
import os
import time
from settings import TIMESTAMP_COL, X_COL, Y_COL, Z_COL, G_UNIT, TIMEZONE

def init_logger(level="DEBUG"):
    numeric_level = getattr(logging, level.upper(), None)
    logging.basicConfig(level=numeric_level)
    
def load_accel_data_from_file(inFile, scale_by_g=False):
    temp = pd.read_csv(inFile, 
            header=0, usecols=[TIMESTAMP_COL, X_COL, Y_COL, Z_COL], 
            dtype={TIMESTAMP_COL:int, X_COL:float, Y_COL:float, Z_COL:float}
    )
    temp[TIMESTAMP_COL] = pd.to_datetime(temp[TIMESTAMP_COL], unit='ms', utc=True).dt.tz_convert(TIMEZONE)
    temp = temp.set_index(TIMESTAMP_COL)
    if scale_by_g:
        temp = temp / G_UNIT
    return temp

def load_all_accelerometer_data(inPath, scale_by_g = False):
    t1 = time.perf_counter()
    
    inFiles = os.listdir(inPath)
    logging.info(f"Loading data from {inPath}")
    logging.info(f"Found {str(len(inFiles))} files")

    #load all raw data, scale by g if phone runs android
    data = pd.concat([load_accel_data_from_file(os.path.join(inPath, f), scale_by_g=scale_by_g) for f in inFiles])
    
    t2 = time.perf_counter()
    logging.info("Finished loading data ({0:8.2f}s)".format(t2-t1))
    return(data)