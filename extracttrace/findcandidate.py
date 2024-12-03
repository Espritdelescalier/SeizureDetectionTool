import re
import pyabf
import pyedflib
from scipy import signal
import pywt
import numpy as np
from extracttrace import paddata
from signalcandidate import SignalCandidate

def decompose_signal(trace, sample_rate, decompLevel=10, factor=5, wavelet='db4'):
    print("sample rate:", sample_rate)
    """
    create notch filter to compare to the other code
    """
    notch_freq = 50
    q_fact = notch_freq / 3.5
    smplrate = sample_rate / factor
   
    b_notch, a_notch = signal.iirnotch(notch_freq, q_fact, sample_rate)
    
    """
    apply notch filter to signal
    """
    fy_notch = signal.filtfilt(b_notch, a_notch, trace[::factor])
    print("Shape of sweepY: {}".format(fy_notch.shape))

    seqlength = fy_notch.shape[0]
    if seqlength % 2 == 1:
        fy_notch = fy_notch[:-1]
        seqlength = fy_notch.shape[0]
        print("New seqlength (should be even) {}".format(fy_notch.shape[0]))
    print("Max decomp level of original data: {}".format(
        pywt.swt_max_level(seqlength)))
    
    """
    Compute padding for the pywavelet modwtmra equivalent
    """
    pad = paddata.pad_data(seqlength, decompLevel)
    print("Pad {}".format(pad))
    fy_notch = np.pad(fy_notch, (pad, pad), 'reflect')
    print("Max decomp level of padded data (boundary effect mitigation -> reflection): {}".format(
        pywt.swt_max_level(fy_notch.shape[0])))
   
    wtmra = pywt.mra(fy_notch, wavelet, transform='swt', mode='periodization')
  
    num_subsequences = decompLevel + 1
    
    return fy_notch, wtmra, pad, seqlength, smplrate, num_subsequences


def detect_anomalies(wtmra, pad, seqlength, smplrate, indices,
                     percentileThresh=90, sizeConvWindow=1000, lowerLimitTime=0.525, 
                     traceAddToAnomaly=5, background=False):
    
    num_sousseq = len(indices) #number of subsequences to analyze
    oneVec = np.ones(sizeConvWindow, dtype=float)
    win = [None] * num_sousseq
    a = [None] * num_sousseq
    wAbovePercentile = [None] * num_sousseq
    
    if background:
        percentileThresh -= 30
        above = 0
        bellow = 1
    else:
        above = 1
        bellow = 0
    
    for i in range(num_sousseq):
        sousseq_index = indices[i] #subsequence index we want
        win[i] = signal.fftconvolve(np.square(wtmra[sousseq_index][pad:pad + seqlength]), oneVec) #fenetre de données transformées
        a[i] = np.percentile(win[i], percentileThresh)
        wAbovePercentile[i] = np.where(win[i] >= a[i], above, bellow)
    
    for i in range(num_sousseq - 1):
        wallAbovePercentile = wAbovePercentile[i] * wAbovePercentile[i + 1]

    indexwall = wallAbovePercentile.nonzero()[0]
    print(indexwall)
    print("indexwall size {}".format(indexwall.shape[0]))

    '''print("Indexes of non zero elements of all boolean array referencing the
    values above the 90th percentile in each win array", indexwall)
    print("shape of the resulting array: {}".format(indexwall.shape))
    '''
    extractedAno = []
    numberofano = 0
    onsettemp = 0
    realonset = 0

    lowerLimitTimeAdjusted = lowerLimitTime * float(smplrate)
    headTailAddToAnomaly = traceAddToAnomaly * smplrate

    for i in range(indexwall.shape[0] - 1):
        if indexwall[i] + 1 != indexwall[i + 1]:
            onsettemp = i + 1
            if i - realonset >= lowerLimitTimeAdjusted:
                extractedAno.append(SignalCandidate(indexwall[realonset], indexwall[i], "wavelet decomposition"))
                numberofano += 1
            realonset = onsettemp

    return extractedAno, headTailAddToAnomaly

def findCandidates(trace, sample_rate, percentileThresh=90, decompLevel=10, sizeConvWindow=1000,
                   indices=[5, 6, 7], lowerLimitTime=0.525, traceAddToAnomaly=5,
                   background=False, wavelet='db4', factor=5):

    fy_notch, wtmra, pad, seqlength, smplrate, num_subsequences = decompose_signal(trace, sample_rate, decompLevel, factor, wavelet)
    extractedAno, headTailAddToAnomaly = detect_anomalies(wtmra, pad, seqlength, smplrate, indices, 
                                                          percentileThresh, sizeConvWindow, 
                                                          lowerLimitTime, traceAddToAnomaly, background)

    return fy_notch, extractedAno, headTailAddToAnomaly, smplrate, num_subsequences

def traces_info_from_file(tracefile, channel=0):
    sep = tracefile.split('.')
    if sep[1] == 'abf':
        print('yay abf')
        f = pyabf.ABF(tracefile)
        smplerate = [f.sampleRate]
        datalength = [f.dataLengthSec]
        trace = [f.sweepY]
        num_of_channels = 1
        eeg_match = 0
        label = ['1']
    elif sep[1] == 'edf' or sep[1] == 'edf+':
        f = pyedflib.EdfReader(tracefile)
        label = f.getSignalLabels()
        print("labels", label)
        trace = []
        smplerate = []
        datalength = []
        eeg_match = 0
        regex = re.compile('(eeg)', re.IGNORECASE)
        for i in range(len(label)):
            m = regex.match(label[i])
            if m:
                print('match found in index', i)
                eeg_match = i
            trace.append(f.readSignal(i))
            smplerate.append(f.getSampleFrequency(i))
            datalength.append(f.getFileDuration())
        num_of_channels = len(trace)
        print('num of channels', num_of_channels)
        f.close()

    return smplerate, datalength, trace, num_of_channels, eeg_match, label
