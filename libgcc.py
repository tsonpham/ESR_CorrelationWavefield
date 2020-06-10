#! /usr/bin/env python

import numpy as np
from obspy import read
from obspy.core import Stream, Trace, UTCDateTime
from obspy.signal.filter import bandpass
from scipy.signal import fftconvolve, hilbert
from glob import glob
from os.path import isdir
from os import mkdir
import time
import pyfftw as fftw
from netCDF4 import Dataset
from obspy.io.sac.sactrace import SACTrace

import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
    
############################################################
########## SUPPORTING  ##########
############################################################

def WriteCorrf(fname, array2d, delta=0.1, bin_size=1.0):
    """
    Save two-dimensional correlogram into a file in NetCDF4 format.
    :param fname: Name of the output NetCDF file.
    :param array2d: Two-dimensional array of correlation wavefield.
    :param delta: Step in the time domain.
    :param bin_size: Step in the inter-receiver distance.
    """
    ## Initiciate the NetCDF4 to record the 2d correlogram.
    rootgrp = Dataset(fname, "w", format="NETCDF4")
    ## Declare Type Dimensions: Inter-receiver distance and correlation lapse time.
    nbins, npts = array2d.shape
    distanceDim = rootgrp.createDimension('distance', nbins)
    timeDim = rootgrp.createDimension('time', npts)
    ## Declare Type Variables: distance, time, corrlogram, delta (time step) and distance bin size
    distanceVar = rootgrp.createVariable('distance', 'f4', ('distance',))
    timeVar = rootgrp.createVariable('time', 'f4', ('time',))
    corrVar = rootgrp.createVariable('corrf', 'f4', ('distance', 'time'), zlib=True)
    deltaVar = rootgrp.createVariable('delta', 'f4')
    binsizeVar = rootgrp.createVariable('bin_size', 'f4')
    # write variables
    distanceVar[:] = (np.arange(nbins) + 0.5) * bin_size
    distanceVar.units = 'degrees'
    timeVar[:] = np.arange(npts) * delta / 60.0
    timeVar.units = 'minutes'
    corrVar[:] = array2d
    deltaVar[:] = delta / 60.0
    deltaVar.units = 'minutes'
    binsizeVar[:] = bin_size
    binsizeVar.units = 'degrees'
    ## Close of NetCDF file writer
    rootgrp.close()
    
def ReadCorrwf(fname):
    """
    Read correlation wavefield from a netCDF file.
    :param fname: Name of NetCDF file containing a correlogram.
    """
    rootgrp = Dataset(fname, "r", format="NETCDF4")
    ## image size
    nbins = len(rootgrp.variables['distance'])
    npts = len(rootgrp.variables['time'])
    imag = np.zeros((nbins, npts))
    ## read in data
    imag[:] = rootgrp.variables['corrf'][:]
    ## read in scalar
    delta = rootgrp.variables['delta'][:]
    bin_size = rootgrp.variables['bin_size'][:]
    ## close file
    rootgrp.close()
    return imag, delta, bin_size

# def SumCorrwf(nc4_list):
#     # retrieve the shape of the correlogram
#     imag, delta, bin_size = read_corrwf(nc4_list[0])
#     nbins, npts = imag.shape
#     npts = int(80/delta)
#     earth_model = nc4_list[0].split('/')[-2]

#     # 1st option to compute the stack
#     imag_stack = np.zeros((len(nc4_list), nbins, npts))
#     hist_bin = 0
#     for n, fname in enumerate(nc4_list):
#        imag, delta, bin_size = ReadCorrwf(fname)
#        hist_bin += imag[:, -1]
#        imag_stack[n, :, :] = imag[:, 0:npts]
#     # calculate mean and standard of deviation
#     imag = np.sum(imag_stack, axis=0)
#     imag[:, -1] = hist_bin
#     return imag, delta, bin_size

def NormalizeFilter(array2d, hist_bin, delta_sec, pmin=15., pmax=50.):
    '''
    Normalize correlogram and bandpass filter.
    :param array2d: 2D correlogram to be normalized.
    :param hist_bin: number of correlation in each inter-receiver distance bin
    :param delta_sec: time step in seconds
    :param pmin: lower bandpass period (in seconds)
    :param pmmax: upper bandpass period (in seconds)
    '''
    ## Getting the dimensions of the correlogram
    nbins, npts = array2d.shape
    ## For each correlogram bin
    for nb in range(nbins): 
        if hist_bin[nb] > 0: 
            ## Normalize the correlation stack by number correlation pair
            data = array2d[nb, :] / hist_bin[nb]
            ## Bandpass filtering 
            array2d[nb, :] = bandpass(data, df=1./delta_sec, freqmin=1./pmax, 
                freqmax=1./pmin, corners=4, zerophase=True)
    return array2d
        
def StackData(stream, order, returnall=False):
    """
    Data stacking methods.
    """
    stack = 0
    if order==0:
        for tr in stream:
            stack += tr.data
        stack /= len(stream)
        return Trace(header={'npts': len(stack), 'delta': stream[0].stats.delta}, data=stack)
    phase = 0j
    for tr in stream:
        stack += tr.data
        ## calculate phase
        asig = hilbert(tr.data)
        phase += asig / np.abs(asig)
    stack /= len(stream)
    weight = np.abs(phase / len(stream))
    if not returnall:
        return Trace(header={'npts': len(stack), 'delta': stream[0].stats.delta}, \
                     data=stack * weight**order)
    else:
        dls = Trace(header={'npts': len(stack), 'delta': stream[0].stats.delta}, \
                    data=stack)
        wt = Trace(header={'npts': len(stack), 'delta': stream[0].stats.delta}, \
                   data=weight)
        pws = Trace(header={'npts': len(stack), 'delta': stream[0].stats.delta}, \
                    data=stack * weight**order)
        return dls, wt, pws

def ReadData(wildcast, origintime, window_start, window_length):
    """
    Read-in coda waveforms in the interested time window.
    """
    fname_list = glob(wildcast)
    fname_list.sort()
    data_stream = Stream()
    for fname in fname_list:
        try:
            tr = read(fname, format='SAC')[0]
            ##########
            if origintime == None: origintime = tr.stats.starttime
            starttime = origintime + window_start
            endtime = starttime + window_length
            tr.trim(starttime, endtime)
            tr.detrend('linear')
            tr.detrend('demean')
            data_stream.append(tr)
        except Exception as ex: 
            continue
            print (ex)
    if len(data_stream) == 0:
        print ('There are NO seismograms coressponding to %s!' % wildcast)
        print ('STOP')
        exit()
    return data_stream
   
def next2(N):
    '''
    Finding the power of 2 that is greater than the positive integer input 'n'.
    '''
    if N <= 0: raise Exception('Invalid non-positive input for next2().')      
    nn = 1
    while(nn < N): nn *= 2
    return nn
##############################################################


## WARNING: Please double think when editing this library since this is extensively used by others!!
## the main function is more flexible 
class GCC:
    '''
    Pre-Processing Worder.
    '''
    def __init__ (self, fft_npts, temp_width, spec_width, ram_fband):
        self.fft_npts = fft_npts
        self.temp_width = temp_width
        self.spec_width = spec_width
        self.ram_fband = ram_fband

        ## prepare FFTW for real input        
        self.re_arr = fftw.empty_aligned(fft_npts, dtype='float32')
        self.rfft = fftw.builders.rfft(self.re_arr)
        # prepare inverse FFT for real input
        self.cp_arr = fftw.empty_aligned(self.get_spec_npts(), dtype='complex64')
        self.irfft = fftw.builders.irfft(self.cp_arr)

    def get_temp_npts(self):
        return self.fft_npts
    
    def get_spec_npts(self):
        return (int(self.fft_npts / 2) + 1)

    def running_absolute_mean(self, trace):
        """
        Running-aboulute-mean normalization.
        """

        # if the temporal normalization weight is not set, return the original trace
        if self.temp_width == None: return
        
        # compute normalizing weight (filter if specified) [Bensen et al., 2007]
        delta = trace.stats.delta
        filtered = trace.data.copy()
        if self.ram_fband != None:
            # filter the original trace to compute weight
            filtered = bandpass(filtered, df=1.0/delta, zerophase=True, \
                    freqmin=self.ram_fband[0], freqmax=self.ram_fband[1])
        
        # smoothen by convolving with an average mask
        winlen = 2 * int(0.5 * self.temp_width / delta) + 1
        avg_mask = np.ones(winlen) / (1.0 * winlen)
        # filtered = fftconvolve(np.abs(filtered), avg_mask, 'same')
        filtered = np.convolve(np.abs(filtered), avg_mask, 'same')

        # except there is near-zero weight
        MAX_AMP = np.max(np.abs(filtered))
        for n in range(trace.stats.npts):
            trace.data[n] = 0 if (filtered[n]<=1e-8*MAX_AMP) else (trace.data[n]/filtered[n])
        trace.taper(type='cosine', max_percentage=0.005)

    def spectral_whitening(self, trace):
        """
        Spectral whitening (or normalization).
        """
        # assign input (real) array and perform forward FFTW
        self.re_arr[:] = 0
        self.re_arr[0:trace.stats.npts] = trace.data
        spectrum = self.rfft()
 
        # return spectrum if 'spec_width' is not set
        if self.spec_width == None: 
            return spectrum

        # compute spectral weight my smoothening        
        freq_delta = 1.0 / (trace.stats.delta * self.fft_npts) # spectral discrete step
        winlen = 2 * int(0.5 * self.spec_width / freq_delta) + 1
        avg_mask = np.ones(winlen) / (1.0 * winlen)
        avg_weight = np.convolve(np.abs(spectrum), avg_mask, 'same')
        MAX_AMP = np.max(np.abs(avg_weight))
        for n in range(len(spectrum)):
            spectrum[n] = 0 if (avg_weight[n]<=1e-8*MAX_AMP) else (spectrum[n]/avg_weight[n])
 
        # return whitened spectrum
        return spectrum

    def pre_proc(self, trace):
        """
        Preprocessing for a single station.
        """
        assert (trace.stats.npts <= self.fft_npts)
        # temporal normalization
        self.running_absolute_mean(trace)
        # spectral normalization (whitening)
        spectrum = self.spectral_whitening(trace)
        
        return spectrum

    def xcorr(self, spec1, spec2, freq_domain=False):
        assert (len(spec1) == len(spec2))
        assert (self.get_spec_npts() == len(spec1))
        
        prod = spec1 * np.conj(spec2)
        
        if freq_domain == True:
            return prod
        else:
            self.cp_arr[:] = prod
            return self.irfft()
        
    def inv_rfft(self, spec):
        assert (len(spec) == self.get_spec_npts())
        
        self.cp_arr[:] = spec
        return self.irfft()