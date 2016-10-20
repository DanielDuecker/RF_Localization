"""This module hacks a DVBT-dongle and abuses it
as a sdr spectrum analyzer between 23 an 1,700 MHz
for underwater RSS based radiolocation purpose.

For more information see:
https://github.com/roger-/pyrtlsdr
http://sdr.osmocom.org/trac/wiki/rtl-sdr
"""

# import modules
from abc import ABCMeta, abstractmethod
import time as t
import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import *
from scipy import signal
from scipy.optimize import curve_fit
from scipy.special import lambertw
from drawnow import *


# connect to sdr
sdr = RtlSdr()
sdr.gain = 1
sdr.sample_rate = 2.048e6

# define classes
class RfEar(object):
    """A simple class to compute PSD with a DVBT-dongle."""

    __metaclass__ = ABCMeta

    def __init__(self, *args):
        """Keyword-arguments:
        *args -- list of frequencies (must be in a range of __sdr.sample_rate)
        """
        self.__freq = []
        self.set_freq(self, *args)
        self.__size = 0
        self.set_size(256)

    def set_size(self, size):
        """Set number of samples to be read by sdr [*1024]."""
        self.__size = size

    def get_size(self):
        """Return number of samples to be read by sdr [*1024]."""
        return self.__size

    def get_iq(self):
        """Read and return self.__size*1024 iq samples at a certain frequency."""
        samples = sdr.read_samples(self.__size * 1024)
        return samples

    def set_freq(self, *args):
        """Defines frequencies where to listen (between 27MMz and 1.7GHz).
        The range must be in the __sdr.sample_rate bandwidth.

        Keyword arguments:
        *args -- single frequency or array of frequencies (default none)
        """
        self.__freq[:] = []
        for arg in args:
            self.__freq.append(arg)
        if not isinstance(self.__freq[0], float):
            self.__freq = self.__freq[1:]
        sdr.center_freq = np.mean(self.__freq)

    def get_freq(self):
        """Returns list of frequencies assigned to the object."""
        return self.__freq

    def set_srate(self, srate):
        """Defines the sampling rate.

        Keyword arguments:
        srate -- samplerate [Ms/s] (default 2.4e6)
        """
        sdr.sample_rate = srate

    def get_srate(self):
        """Returns sample rate assigned to object
        and gives default tuner value.
        range is between 1.0 and 3.2 MHz
        """
        print ('Default sample rate: 2.4MHz')
        print ('Current sample rate: ' + str(sdr.sample_rate))

    def plot_psd(self):
        """Get Power Spectral Density Live Plot."""
        sdr.center_freq = np.mean(self.__freq)
        plt.ion()      # turn interactive mode on
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                plt.clf()
                plt.axis([-1.5e6,
                          1.5e6, -120, 0])
                samples = self.get_iq()
                # use matplotlib to estimate and plot the PSD
                freq, pxx_den = signal.periodogram(samples,
                                                   fs=sdr.sample_rate, nfft=1024)
                plt.plot(freq, 10*np.log10(pxx_den)) #rss in dB
                xlabels = np.linspace((sdr.center_freq-.5*sdr.sample_rate)/1e6,
                                      (sdr.center_freq+.5*sdr.sample_rate)/1e6, 5)
                plt.xticks(np.linspace(min(freq), max(freq), 5), xlabels, rotation='horizontal')
                plt.grid()
                plt.xlabel('Frequency [MHz]')
                plt.ylabel('Power [dB]')
                plt.show()
                plt.pause(0.001)
            except KeyboardInterrupt:
                plt.show()
                print ('Liveplot interrupted by user')
                drawing = False
        return pxx_den, freq

        return freq, pxx_den

    def get_rss(self):
        """Find maximum power values around specified freq points.
        Returns received signal strength (rss) in dB.
        """
        samples = self.get_iq()
        freq, pxx_den = signal.periodogram(samples,
                                           fs=sdr.sample_rate, nfft=1024)
        del freq
        if len(self.__freq) == 1:
            rss = [10*np.log10(max(pxx_den))] #Power in dB !!!
        elif len(self.__freq) == 2:
            pxx_den_left = pxx_den[:len(pxx_den)/2]
            pxx_den_right = pxx_den[len(pxx_den)/2:]
            rss = [10*np.log10(max(pxx_den_left)),
                10*np.log10(max(pxx_den_right))]
        return rss


    def get_absfreq_pden_sorted(self):
        """
        gets the iq-samples and calculates the powerdensity.
        It sorts the freq and pden vector such that the frequencies are increasing.
        Moreover the centerfrequency is added to freq such that freq contains the absolute frequencies for the
        corresponding powerdensity

        returns freqsort, pxx_densort
        """
        samples = self.get_iq()
        freq, pxx_den = signal.periodogram(samples,
                                           fs=sdr.sample_rate, nfft=1024)

        freq_sorted = np.concatenate((freq[len(freq) / 2:], freq[:len(freq) / 2]), axis=1)
        pxx_den_sorted = np.concatenate((pxx_den[len(pxx_den) / 2:], pxx_den[:len(pxx_den) / 2]), axis=1)

        freq_sorted = freq_sorted + sdr.center_freq # add centerfreq to get absolut frequency values


        return freq_sorted, pxx_den_sorted

    def get_max_rss_in_freqspan(self, freqset, freqspan, freq, pxx_den):
        """
        find maximum rss peaks in spectrum
        :param freqset: frequency which max power density is looked for
        :param freqspan: width of the frequency span
        :param freq: vector of input frequency, must have increasing frequencies
        :param pxx_den: vector of power spectrum
        :return: frequeny, maxpower
        """
        startindex = 0
        endindex = len(freq)
        i = 0
        while i < len(freq):
            if freq[i] >= freqset - freqspan / 2:
                startindex = i
                break
            i = i + 1

        while i < len(freq):
            if freq[i] >= freqset + freqspan / 2:
                endindex = i
                break
            i = i + 1

        pxx_den = np.array(pxx_den)

        # find index of the highest power density
        maxind = np.where(pxx_den == max(pxx_den[startindex:endindex]))

        pxx_den_max = pxx_den[maxind]
        freq_den_max = freq[maxind]

        return freq_den_max, pxx_den_max



    def plot_multi_rss_live(self, freq1, freq2, freqspan = 2e5, numofsamples = 250):
        """

        :param freq1: 1st frequency to track the power peak
        :param freq2: 2nd frequency to track the power peak
        :param freqspan: width of the frequencyspan around the tracked frq
        :param numofsamples: number of displayed samples (default= 250)
        :return: rss1, rss2
        """
        plt.ion()      # turn interactive mode on
        drawing = True
        cnt = 0
        rss1 = []
        rss2 = []

        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                freq, pxx_den = self.get_absfreq_pden_sorted()

                # find maximum power peaks in spectrum
                freq_found1, pxx_den_max1 = self.get_max_rss_in_freqspan(freq1, freqspan, freq, pxx_den)
                freq_found2, pxx_den_max2 = self.get_max_rss_in_freqspan(freq2, freqspan, freq, pxx_den)


                #print("max power = " + str(pxx_den_max) + " at freq = " + str(freq_den_max))

                # plotting
                rss1.append(10 * np.log10(pxx_den_max1))
                rss2.append(10 * np.log10(pxx_den_max2))

                plt.clf()
                plt.title("Live Streaming RSS-Values")
                plt.ylim(-120,0)
                plt.plot(rss1, 'b.-', label="Freq1 = " + str(freq1/ 1e6) + ' MHz' + " @ " +str(freq_found1 / 1e6) + ' MHz')
                plt.plot(rss2, 'r.-', label="Freq2 = " + str(freq2/ 1e6) + ' MHz' + " @ " +str(freq_found2 / 1e6) + ' MHz')  # rss in dB
                #plt.axis([-1.5e6,
                #          1.5e6, -120, 0])
                plt.ylabel('Power [dB]')
                plt.grid()
                plt.legend(loc='lower right')
                #plt.show()
                plt.pause(0.001)
                cnt = cnt + 1
                if cnt > numofsamples:
                    rss1.pop(0)
                    rss2.pop(0)

            except KeyboardInterrupt:
                plt.show()
                print ('Liveplot interrupted by user')
                drawing = False


        # return frequency with highest power and its power density
        return rss1, rss2

    @abstractmethod
    def rfear_type(self):
        """Return a string representing the type of rfear this is."""
        pass

    #unused
    def rpi_get_power(self, printing=0, size=256):
        """Routine for Raspberry Pi.

        Keyword arguments:
        printing -- visible output on terminal (default  0)
        size -- measure for length of fft (default 256*1024)
        """
        sdr.center_freq = np.mean(self.__freq)
        running = True
        pmax = []
        self.set_size(size)
        while running:
            try:
                pmax.append(self.get_rss())
                if printing:
                    print (self.get_rss())
                    print ('\n')
                else:
                    pass
            except KeyboardInterrupt:
                print ('Process interrupted by user')
                return pmax

class CalEar(RfEar):
    """Subclass of Superclass RfEar for modelling and testing purpose."""
    def __init__(self, *args):
        RfEar.__init__(self, *args)

    def plot_rss(self, time=10.0):
        """Measures RSS [dB] of specified frequencies
        for a certain time.

        Keyword arguments:
        time -- time of measurement [s] (default  10.0)
        """
        sdr.center_freq = np.mean(self.get_freq())
        powerstack = []
        elapsed_time = 0
        timestack = []
        while elapsed_time < time:
            start_calctime = t.time()
            powerstack.append(self.get_rss())
            t.sleep(0.005)
            calctime = t.time() - start_calctime
            timestack.append(calctime)
            elapsed_time = elapsed_time + calctime
            calctime = np.mean(calctime)
        plt.clf()
        plt.grid()
        plt.axis([0, len(powerstack), -120, 10])
        powerstack = np.array(powerstack)
        for i in range(len(self.get_freq())):
            plt.plot(powerstack[:, i], 'o', label=str(self.get_freq()[i]/1e6)+' MHz')
        plt.legend(loc='upper right')
        plt.xlabel('Updates')
        plt.ylabel('Maximum power (dB)')
        plt.show()
        return powerstack



    def make_test(self, time=10.0):
        """Interactive method to get PSD data
        at characteristic frequencies.

        Keyword arguments:
        time -- time of measurement [s] (default 10.0)
        """
        sdr.center_freq = np.mean(self.get_freq())
        testing = True
        modeldata = []
        variance = []
        plt.figure()
        plt.grid()
        while testing:
            try:
                raw_input('Press Enter to make a measurement,'
                          ' or Ctrl+C+Enter to stop testing:\n')
                elapsed_time = 0
                powerstack = []
                print (' ... measuring ...')
                while elapsed_time < time:
                    start_calctime = t.time()
                    powerstack.append(self.get_rss())
                    calc_time = t.time() - start_calctime
                    elapsed_time = elapsed_time + calc_time
                    t.sleep(0.01)
                print ('done\n')
                t.sleep(0.5)
                print (' ... evaluating ...')
                modeldata.append(np.mean(powerstack))
                variance.append(np.var(powerstack))
                plt.clf()
                plt.errorbar(range(len(modeldata)), modeldata, yerr=variance,
                             fmt='o', ecolor='g')
                plt.xlabel('Evaluations')
                plt.ylabel('Mean maximum power [dB]')
                plt.grid()
                plt.show()
                del powerstack
                print ('done\n')
                t.sleep(0.5)
            except KeyboardInterrupt:
                print ('Testing finished')
                testing = False
        return modeldata, variance

    def measure_rss_var(self, freqset, freqrange=2e4, time=10.0):
        """
        Interactive method to get PSD data
        at characteristic frequencies.
        :param freqset: tx-frequency [Hz]
        :param freqrange: range [Hz] around tx-frequency where the peak-rss lies in
        :param time: time of measurement [s] (default 10.0)
        :return: modeldata, variance - arrays with mean rss for each distance + its variance
        """
        sdr.center_freq = np.mean(self.get_freq())
        testing = True
        modeldata = []
        variance = []
        plt.figure()
        plt.grid()
        print ('RSS ist measured at freq: ' + str(freqset / 1e6) +
               'MHz, frequency span is +/-' + str(freqrange / 1e3) + 'kHz \n')
        while testing:
            try:
                raw_input('Press Enter to make a measurement,'
                          ' or Ctrl+C+Enter to stop testing:\n')
                elapsed_time = 0
                powerstack = []
                print (' ... measuring for ' + str(time) + 's ...')
                while elapsed_time < time:
                    start_calctime = t.time()
                    freq_sort, pxx_den_sort = self.get_absfreq_pden_sorted()  # get samples and sort freq/power-array
                    powerstack.append(self.get_max_rss_in_freqspan(freqset, freqrange, freq_sort, pxx_den_sort))
                    calc_time = t.time() - start_calctime
                    elapsed_time = elapsed_time + calc_time
                    t.sleep(0.01)
                print ('done\n')
                t.sleep(0.5)
                print (' ... evaluating ...')
                modeldata.append(np.mean(powerstack))
                variance.append(np.var(powerstack))
                plt.clf()
                plt.errorbar(range(len(modeldata)), modeldata, yerr=variance,
                             fmt='o', ecolor='g')
                plt.xlabel('Evaluations')
                plt.ylabel('Mean maximum power [dB]')
                plt.grid()
                plt.show()
                del powerstack
                print ('done\n')
                t.sleep(0.5)
            except KeyboardInterrupt:
                print ('Testing finished')
                testing = False

        return modeldata, variance

    def get_performance(self, bandwidth=2.4e6):
        """Measure performance at certain sizes and sampling rates.

        Keyword arguments:
        bandwidth -- sampling rate of sdr [Ms/s] (default 2.4e6)
        """
        sdr.center_freq = np.mean(self.get_freq())
        self.set_srate(bandwidth)
        measurements = 100
        SIZE = [4, 8, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
        VAR = []
        MEAN = []
        UPDATE = []
        total_time = 0
        for i in SIZE:
            cnt = 0
            powerstack = []
            timestack = []
            elapsed_time = 0
            while cnt <= measurements:
                cnt = cnt+1
                start_calctime = t.time()
                # use matplotlib to estimate the PSD and save the max power
                self.set_size(i)
                powerstack.append(self.get_rss())
                t.sleep(0.005)
                calctime = t.time() - start_calctime
                timestack.append(calctime)
                elapsed_time = elapsed_time + calctime
            calctime = np.mean(timestack)
            VAR.append(np.var(powerstack))
            MEAN.append(np.mean(powerstack))
            UPDATE.append(calctime)
            total_time = total_time+elapsed_time
        print ('Finished.')
        print ('Total time [sec]: ')
        print (total_time)
        plt.figure()
        plt.grid()
        plt.plot(SIZE, VAR, 'ro')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Variance (dB)')
        plt.figure()
        plt.grid()
        plt.errorbar(SIZE, MEAN, yerr=VAR,
                     fmt='o', ecolor='g')
        plt.plot(SIZE, MEAN, 'x')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Mean Value (dB)')
        plt.figure()
        plt.grid()
        plt.plot(SIZE, UPDATE, 'g^')
        plt.xlabel('Sample Size (*1024)')
        plt.ylabel('Update rate (sec)')
        plt.show()
        return SIZE, VAR, MEAN, UPDATE

    def get_model(self, pdata, vdata):
        """Create a function to fit with measured data.
        alpha and xi are the coefficients that curve_fit will calculate.
        The function structure is known.

        Keyword arguments:
        :param pdata -- array containing the power values [dB]
        :param vdata -- array containing the variance of the measurement series [dB]
        """
        x_init = raw_input('Please enter initial distance [cm]: ')
        x_step = raw_input('Please enter step size [cm]:')
        xdata = np.arange(int(x_init), int(x_init)+len(pdata)*int(x_step), int(x_step))
        xdata = np.array(xdata, dtype=float)
        pdata = np.array(pdata, dtype=float)
        vdata = np.array(vdata, dtype=float)
        plt.figure()
        plt.grid()
        plt.errorbar(xdata, pdata, yerr=vdata,
                     fmt='ro', ecolor='g', label='Original Data')
        def func(xdata, alpha, xi):
            """Range Sensor Model (RSM) structure."""
            return -20*np.log10(xdata)-alpha*xdata-xi
        popt, pcov = curve_fit(func, xdata, pdata)
        del pcov
        print ('alpha = %s , xi = %s' % (popt[0], popt[1]))
        xdata = np.linspace(xdata[0], xdata[-1], num=1000)
        plt.plot(xdata, func(xdata, *popt), label='Fitted Curve')
        plt.legend(loc='upper right')
        plt.xlabel('Distance [cm]')
        plt.ylabel('RSS [dB]')
        plt.show()
        return popt

    def rfear_type(self):
        """"Return a string representing the type of rfear and its properties."""
        print ('CalEar,')
        print ('Tuned to:' + str(self.get_freq()) + ' MHz,')
        self.get_srate()
        print ('Reads ' + str(self.get_size()) + '*1024 8-bit I/Q-samples from SDR device.')

class LocEar(RfEar):
    """Subclass of Superclass RfEar for 2D dynamic object localization."""
    def __init__(self, alpha, xi, *args):
        RfEar.__init__(self, *args)
        self.__alpha = alpha
        self.__xi = xi

    def calibrate(self):
        """Adjust RSM in line with measurement."""
        d_ref = raw_input('Please enter distance'
                          'from transmitter to receiver [cm]: ')
        def func(ref, xi_cal):
            """RSM structure with correction param xi_cal."""
            return -20*np.log10(ref[0])-ref[1]*ref[0]-ref[2]+xi_cal
        time = 5.0
        elapsed_time = 0
        powerstack = []
        print (' ... measuring ...')
        while elapsed_time < time:
            start_calctime = t.time()
            powerstack.append(self.get_rss())
            calc_time = t.time() - start_calctime
            elapsed_time = elapsed_time + calc_time
            t.sleep(0.01)
        print ('done\n')
        t.sleep(0.5)
        print (' ... evaluating ...')
        powerstack = np.array(powerstack)
        p_mean = []
        for i in range(len(self.get_freq())):
            p_mean.append(np.mean(powerstack[:, i]))
        freq_ref = np.argmax(p_mean)
        print ('Variance [dB]:')
        print (np.var(powerstack[:, freq_ref]))
        print ('Calibration reference frequency: ' + str(self.get_freq()[freq_ref]/1e6) + ' MHz')
        d_ref = np.array(d_ref, dtype=float)
        p_ref = np.array(max(p_mean), dtype=float)
        popt, pcov = curve_fit(func, [d_ref, self.__alpha, self.__xi], p_ref)
        del pcov
        print ('Xi alt:')
        print (self.__xi)
        self.__xi = self.__xi+popt[0]
        print ('Xi neu:')
        print (self.__xi)

    def get_caldata(self):
        """Returns the calibrated RSM params."""
        return self.__alpha, self.__xi

    def map_path(self, d_t=55.0):
        """Maps estimated location in 1D or 2D respectively.

        Keyword argumennts:
        d_t -- distance between the transmitting stations [cm] (default 55.0)
        """
        sdr.center_freq = np.mean(self.get_freq())
        x_min = -10.0
        x_max = d_t+10.0
        y_min = -100.0
        y_max = 100.0
        plt.axis([x_min, x_max, y_min, y_max])
        plt.ion()
        plt.grid()
        plt.xlabel('x-Axis [cm]')
        plt.ylabel('y-Axis [cm]')
        drawing = True
        pos_est = []
        try:
            while drawing:
                rss = self.get_rss()
                pos_est.append(self.lambertloc(rss))
                if len(pos_est[-1]) == 1:
                    plt.plot(pos_est[-1], 0, 'bo')
                elif len(pos_est[-1]) == 2:
                    x_est = (pos_est[-1][0]**2-pos_est[-1][1]**2+d_t**2)/(2*d_t)
                    y_est = np.sqrt(pos_est[-1][0]**2 - x_est**2)
                    print ([x_est, y_est])
                    plt.plot(x_est, y_est, 'bo')
                plt.show()
                plt.pause(0.001)
                print (pos_est[-1])
                print ('\n')
        except KeyboardInterrupt:
            print ('Localization interrupted by user')
            drawing = False
        return pos_est

    def lambertloc(self, rss):
        """Inverse function of the RSM. Returns estimated range in [cm].

        Keyword arguments:
        rss -- received power values [dB]
        alpha, xi -- RSM params
        """
        Z = [20/(np.log(10)*self.__alpha)*lambertw(np.log(10)*self.__alpha/20*np.exp(-np.log(10)/20*(i+self.__xi))) for i in rss]
        return [z.real for z in Z]

    def rfear_type(self):
        """"Return a string representing the type of rfear this is."""
        print ('LocEar,')
        print ('Alpha: ' + str(self.__alpha) + ', Xi: ' + str(self.__xi))
        print ('Tuned to:' + str(self.get_freq()) + ' MHz,')
        self.get_srate()
        print ('Reads ' + str(self.get_size()) + '*1024 8-bit I/Q-samples from SDR device.')

# define general methods
def plot_result(results):
    """Plot results extracted from textfile."""
    plt.figure()
    plt.grid()
    plt.axis([0, len(results), -50, 30])
    plt.plot(10*np.log10(results), 'o')
    plt.xlabel('Updates')
    plt.ylabel('Maximum power (dBm)')
    plt.show()

def write_to_file(results, text, filename='Experiments'):
    """Save experimental results in a simple text file.

    Keyword arguments:
    results -- list containing data
    text -- description of results
    filename -- name of file (default 'Experiments')
    """
    datei = open(filename, 'a')
    datei.write(t.ctime() + '\n')
    datei.write(text + '\n')
    datei.write(str(results))
    datei.write('\n\n')
    datei.close()

def plot_map(pos_est, x_max=53.5):
    """Plot path from recorded data.

    Keyword arguments:
    x_max -- distance between the transmitting stations [cm] (default 53.5)
    """
    x_min = 0
    y_min = -100
    y_max = 100
    x_est = []
    y_est = []
    for i in pos_est:
        x_est.append((np.power(i[0],2)-np.power(i[1],2)+np.power(x_max,2))/(2*x_max))
        y_est.append(np.sqrt(np.power(i[0],2) - np.power(x_est[-1],2)))
    plt.figure()
    plt.plot(x_est, y_est, 'bo')
    plt.axis([x_min, x_max, y_min, y_max])
    plt.grid()
    plt.show()




