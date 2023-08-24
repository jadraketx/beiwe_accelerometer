import argparse
import logging
import os
import sys
import time
import pandas as pd

from utils import init_logger
from settings import TIME,ACC_BIN_SIZE, TIMEZONE, VALID_DAY_START, VALID_DAY_END, EXERT_ACTIVITY_CUTOFF_STD

def get_sampling_metadata(day, df):

    daytime = df.between_time(start_time=VALID_DAY_START, end_time=VALID_DAY_END, inclusive='left')
    daytime = daytime[daytime['nsamples']>1]

    res = {
        "day": day,
        "num_epochs_obs_all": len(df),
        "num_epochs_wear_all": sum(df['non_wear'] == 0),
        "num_epochs_non_wear_all": sum(df['non_wear'] == 1),
        "num_epochs_missing_all": (1440 - len(df)),
        "mean_samples_per_epoch_all": df['nsamples'].mean(),
        "min_samples_per_epoch_all": df['nsamples'].min(),
        "max_samples_per_epoch_all": df['nsamples'].max(),
        "sd_samples_per_epoch_all": df['nsamples'].std(),
        "coverage_all": (len(df)*100/1440),
        "num_epochs_obs_day": len(daytime),
        "num_epochs_wear_day": sum(daytime['non_wear'] == 0),
        "num_epochs_non_wear_day": sum(daytime['non_wear'] == 1),
        "num_epochs_missing_day": (720 - len(daytime)),
        "mean_samples_per_epoch_day": daytime['nsamples'].mean(),
        "min_samples_per_epoch_day": daytime['nsamples'].min(),
        "max_samples_per_epoch_day": daytime['nsamples'].max(),
        "sd_samples_per_epoch_day": daytime['nsamples'].std(),
        "coverage_day": (len(daytime)*100/720)
    }

    return(res)

def get_physical_activity_metrics(d, df):
    '''Computes various physical activity metrics during daytime hours'''
    daytime = df.between_time(start_time=VALID_DAY_START, end_time=VALID_DAY_END, inclusive='left')
    df_wear = daytime[daytime['non_wear'] == 0]
    min_wear = len(df_wear)

    #frac exertional
    sum_var = df_wear['xStd']**2 + df_wear['yStd']**2 + df_wear['zStd']**2
    is_exert = sum_var > EXERT_ACTIVITY_CUTOFF_STD
    prop_exert = sum(is_exert) / len(is_exert)


    #average enmo
    enmo = df_wear['enmoMean']
    enmo = enmo.sort_values(ascending=False)
    enmoMean = enmo.mean()
    enmoVar = enmo.var()

    #average mad
    madMean = df_wear['mad'].mean()

    #mx metrics
    m5=m10=m15=m30=m60=m60=m5=m90=m120=-1
    l = len(enmo)
    if l>=5:
        m5 = enmo[5-1]
    if l>=10:
        m10 = enmo[10-1]
    if l>=15:
        m15 = enmo[15-1]
    if l>=30:
        m30 = enmo[30-1]
    if l>=60:
        m60 = enmo[60-1]
    if l>=90:
        m90 = enmo[90-1]
    if l>=120:
        m120 = enmo[120-1]

    res = {
        "day":d,
        "prop_exert": prop_exert,
        "avg_enmo": enmoMean,
        "var_enmo": enmoVar,
        "avg_mad": madMean,
        "m5":m5, "m10":m10, "m15":m15, "m30":m30, "m60":m60, "m90":m90, "m120":m120,
        "wear_minutes": min_wear

    }

    return(res)
   

def compute_summaries(params):
    
    logging.info("--------COMPUTING DAILY SUMMARIES----------")

    #load epoch data
    wd = params.workdir
    id = params.id

    #assume path = [wd]/processed_accel/epoch_features/[id]/[id]_epoch_features.csv
    epoch_data_path = os.path.join(wd,"processed_accel","epoch_features_2",id,id+"_epoch_features.csv")

    if not os.path.exists(epoch_data_path):
        logging.error(f"File {epoch_data_path} not found.")
        sys.exit(1)
    
    epoch_data = pd.read_csv(epoch_data_path)
    epoch_data[TIME] = pd.to_datetime(epoch_data[TIME], utc=True).dt.tz_convert(TIMEZONE)
    epoch_data = epoch_data.set_index(TIME)
    daily = epoch_data.resample('1d')

    num_days = len(daily)
    sampling_stats = [0]*num_days
    daily_features = [0]*num_days
    i = 0
    for d, df in daily:
        n = len(df)
        if n == 0:
            logging.info(f"{d} has no observation")
            continue
        logging.info(f"Processing features for {d} ({i}/{num_days})")
        sampling_stats[i] = get_sampling_metadata(d, df)
        daily_features[i] = get_physical_activity_metrics(d,df)
        i = i + 1
    sampling_stats = pd.DataFrame(sampling_stats)
    daily_features = pd.DataFrame(daily_features)

    outdir = os.path.join(wd,"processed_accel","daily_features_3")
    logging.info(f"Setting output directory to {outdir}")
    os.makedirs(outdir, exist_ok=True)
    user_dir = os.path.join(outdir, id)
    os.makedirs(user_dir, exist_ok=True)

    sampling_out = os.path.join(user_dir, id+"_sampling_stats.csv")
    daily_features_out = os.path.join(user_dir, id+"_daily_features.csv")

    logging.info(f"Writing sampling stats to {sampling_out}")
    logging.info(f"Writing daily features to {daily_features_out}")
    sampling_stats.to_csv(sampling_out, index=False)
    daily_features.to_csv(daily_features_out, index=False)

    #compute coverage across minutes of day - collapsed
    #first construct list of minutes in a day
    logging.info("Calculating sampling coverage across minutes of day")
    start = list(daily.groups)[0]
    end = start + pd.Timedelta(1,"day")
    minutes = pd.date_range(start,end, freq="1T")
    minutes = [d.strftime('%H:%M:%S') for d in minutes]
    res = []
    for i in range(0,len(minutes)-1):
        temp = epoch_data.between_time(minutes[i], minutes[i+1], inclusive='left')
        num_minutes = 0
        avg_samples_per_minute = 0
        sd_samples_per_minute = 0
        min_samples_per_minute = 0
        max_samples_per_minute = 0
        if len(temp) > 0:
            num_minutes = len(temp)
            avg_samples_per_minute = temp['nsamples'].mean()
            sd_samples_per_minute = temp['nsamples'].std()
            min_samples_per_minute = temp['nsamples'].min()
            max_samples_per_minute = temp['nsamples'].max()
        res.append({
            'minute':minutes[i],
            'count':num_minutes,
            'avg_samples': avg_samples_per_minute,
            'sd_samples': sd_samples_per_minute,
            'min_samples': min_samples_per_minute,
            'max_samples':max_samples_per_minute
        })
        
    #counts = [len(epoch_data.between_time(minutes[i], minutes[i+1], inclusive='left')) for i in range(0,len(minutes)-1)]
    
    coverage = pd.DataFrame(res)
    coverage_out = os.path.join(user_dir, id+"_minute_coverage.csv")
    logging.info(f"Writing minute coverage to {coverage_out}")
    coverage.to_csv(coverage_out, index=False)

    logging.info("------------COMPLETE-------------")


    

if __name__ == "__main__":

    init_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', required=True, type=str)
    parser.add_argument('--id', required=True,type=str)

    args = parser.parse_args()

    #args = {
    #    'workdir':'/Users/jadrake/Local_dev/beiwe_msk_analysis',
    #    'id':'g5xnzgjd',
    #}

    compute_summaries(args)
    