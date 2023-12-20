#features
VALID_DAY_START = "08:00:00" #8am
VALID_DAY_END = "20:00:00" #8pm
NUM_EPOCHS_IN_VALID_DAY = 721
EXERT_ACTIVITY_CUTOFF_STD = 0.15 #g^2
ACC_BIN_SIZE=1 #g
G_UNIT = 9.80665 #m/s^2
EPOCH_SIZE = 60 #seconds

#Calibration settings
STATIONARY_CUTOFF = 0.003 #g
STATIONARY_EPOCH_SIZE = '60s' #pandas resample format
MAXITER = 1000
IMPROV_TOL = 0.0001  # 0.01%
ERR_TOL = 0.01  # 10mg
CALIB_CUBE = 0.3
CALIB_MIN_SAMPLES = 50

#wear/non-wear paramters
#STATIONARY_CUTOFF = 0.004
NON_WEAR_WINDOW = 30 #minutes


#Raw data
TIMESTAMP_COL = "timestamp"
TIME="time"
X_COL = "x"
Y_COL = "y"
Z_COL = "z"
TIMEZONE = "America/Chicago"
