'''
everything will be written in C-style code rather than optimized Python
functions... this is so it's easier to port to the M4 target


good reference for frequencies
https://pages.mtu.edu/~suits/notefreqs.html 

we should do a coarse search, then a fine search

consider LPF with cutoff at 1400Hz to reduce harmonics

consider different similarity metrics in difference function (sqrd, abs, etc)

average a few periods to get a better estimate of frequency

### below was for testing... generate a clean wave ###
# generate a clean sine wave at low E (82.41 Hz)
freq = 82.41
ang_freq = 2*np.pi*freq
x = np.arange(266688)
#full_file= np.sin(ang_freq*x / fs)

'''

import numpy as np
from scipy.io.wavfile import read
from matplotlib import pyplot as plt


def difference(tau, w, sample):
    d = 0.0
    for i in np.arange(w):
        d = d + np.abs(sample[i] - sample[i+w+tau])
    return d

def get_data(index, w, sample):
    return sample[index:index+3*w]

def zero_crossing(signal):
    
    thresh = 10
    n = signal.shape[0]
    indicator = np.zeros(n)

    for i in np.arange(thresh, n-thresh):
        if ((signal[i] > 0) and (signal[i+1] < 0)):
            
            # simple average denoising... cross to negative
            if ((np.mean(signal[i-thresh:i]) > 0) and (np.mean(signal[i:i+thresh]) < 0)):
                indicator[i] = -1
                
        elif((signal[i] < 0) and (signal[i+1] > 0)):
            
            # simple average denoising... cross to positive
            if ((np.mean(signal[i-thresh:i]) < 0) and (np.mean(signal[i:i+thresh]) > 0)):
                indicator[i] = 1

    return indicator

def get_cross_index(crossings, polarity, offset=0):
    n = crossings.shape[0]

    # looking for the first 1    
    if (polarity == 'to_pos'):
        for i in np.arange(offset, n):
            if (crossings[i] == 1):
                return i

    # looking for the first -1    
    elif (polarity == 'to_neg'):
        for i in np.arange(offset, n):
            if (crossings[i] == -1):
                return i
    
    else:
        print("invalid polarity")
        return 0

# find minimum 
def get_tone(signal, fs):
    
    # subtract average from diff signal
    mu = np.mean(signal)
    signal = signal - .4*mu
    
    # get average time between zero crossings
    indicator = zero_crossing(signal)
        
    # find first to negative crossing, c1
    c1 = get_cross_index(indicator, 'to_neg')
    
    # find first to positive crossing, c2
    c2 = get_cross_index(indicator, 'to_pos', c1)

    # find minimum in trough, m1
    m1 = np.argmin(signal[c1:c2]) + c1
    
    # find second to negative crossing, c3
    c3 = get_cross_index(indicator, 'to_neg', c2)
    
    # find second to positive crossing, c4
    c4 = get_cross_index(indicator, 'to_pos', c3)
    
    # find minimum in trough, m2
    m2 = np.argmin(signal[c3:c4]) + c3
    
    # calculate period of signal
    T = (m2 - m1) / fs
    
    return 1.0/T


# select a saved note to test it on
filename = 'b1.wav'

# read file
fs, aud = read(filename)
if (aud.ndim > 1):
    full_file = aud[:, 0].astype(float)
else:
    full_file = aud

# put data in 5V #10 bit range (10 bit ADC is pretty fast)
max_val = 5 #2**10 - 1
sample_max = np.max(full_file)
sample_min = np.min(full_file)
full_file = max_val * (full_file - sample_min) / (sample_max - sample_min)

# lowest frequency on bass: A_0 is 27.50Hz
# highest note on guitar: E_6 is 1318.51Hz
lowest_freq = 27
highest_freq = 1319
num_periods = 1 # sample for 1 periods of the lowest note
delta_c = 1 # 1Hz steps for coarse search

w_time = num_periods / lowest_freq             # seconds in window
w = int(w_time * fs)                           # number of samples in window
coarse_lag_size = delta_c * w                  # steps for tau
coarse_diff = np.zeros([coarse_lag_size, 1])

# get data sample
sample = get_data(20000, w, full_file)
for i in np.arange(coarse_lag_size):
    coarse_diff[i] = difference(i, w, sample)


# plt.plot(coarse_diff) # if you want to see the signal ...
print('==============================')
print('==============================')
print('estimated frequency: ', round(get_tone(coarse_diff, fs), 2), 'Hz')
print('now do a table look up to get the corresponding note')
