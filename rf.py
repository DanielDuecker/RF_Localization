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


# define classes
class RfEar(object):
    """A simple class to compute PSD with a DVBT-dongle."""
    def __init__(self, center_freq, freqspan=1e5):
        """
        init-method
        :param center_freq: [Hz] Defines the center frequency where to listen (between 27MHz and 1.7GHz)
        :param freqspan: [Hz] span within the the algorithm is looking for amplitude peaks
        """
        # connect to sdr
        self.__sdr = RtlSdr()
        self.__sdr.gain = 1
        self.__sdr.sample_rate = 2.048e6  # 2.048 MS/s

        self.__centerfreq = center_freq
        self.set_sdr_centerfreq(self.__centerfreq)

        self.__freqspan = freqspan
        self.__samplesize = 32

        self.__btxparamsavailable = False
        self.__freqtx = 0
        self.__numoftx = 0
        self.__txpos = []

        self.__bcalparamsavailable = False
        self.__txalpha = []
        self.__txgamma = []

    def get_freqspan(self):
        return self.__freqspan

    def check_txparamsavailable(self):
        return self.__btxparamsavailable

    def set_txparams(self, freqtx, txpos):
        self.__freqtx = freqtx
        self.__numoftx = len(freqtx)
        self.__txpos = txpos

        self.__btxparamsavailable = True
        return self.__btxparamsavailable

    def get_txparams(self):
        """

        :return: self.__freqtx, self.__numoftx, self.__txpos
        """
        if self.__btxparamsavailable:
            return self.__freqtx, self.__numoftx, self.__txpos

    def check_calparamsavailable(self):
        return self.__bcalparamsavailable

    def set_calparams(self, txalpha, txgamma):
        """
        set tx parameters which were obtained
        :param txalpha:
        :param txgamma:
        :return:
        """

        if self.__numoftx == len(txalpha):
            self.__txalpha = txalpha
        else:
            print('ERROR - Setting tx calibration parameter')
            print('txalpha = ' + str(txalpha))
            print('Number of tx is ' + str(self.__numoftx))
            print('Number of txalpha is' + str(len(txalpha)))

            self.__bcalparamsavailable = False
            return self.__bcalparamsavailable

        if self.__numoftx == len(txgamma):
            self.__txgamma = txgamma
        else:
            print('ERROR - Setting tx calibration parameter')
            print('txgamma = ' + str(txgamma))
            print('Number of tx is ' + str(self.__numoftx))
            print('Number of txgamma is ' + str(len(txgamma)))

            self.__bcalparamsavailable = False
            return self.__bcalparamsavailable

        self.__bcalparamsavailable = True
        return self.__bcalparamsavailable

    def set_samplesize(self, samplesize):
        """Set number of samples to be read by sdr [*1024].
        Keyword arguments:
        :param samplesize -- size of the samples be read by sdr [*1024]
        """
        self.__samplesize = samplesize

    def get_samplesize(self):
        """Return number of samples to be read by sdr [*1024]."""
        return self.__samplesize

    def get_sdr_iq_sample(self):
        """ Reads the I/Q-samples from the sdr-dongle

        :return: self.__samplesize*1024 iq samples around the center frequency.
        """
        iq_sample = self.__sdr.read_samples(self.__samplesize * 1024)
        return iq_sample

    def set_sdr_centerfreq(self, centerfreq):
        """Defines the center frequency where to listen (between 27MHz and 1.7GHz).
        The range must be in the __sdr.sample_rate bandwidth.

        Keyword arguments:
        :param centerfreq -- [Hz] single frequency
        """
        self.__sdr.center_freq = centerfreq

    def get_sdr_centerfreq(self):
        """

        :return: sdr center frequency [Hz]
        """
        return self.__sdr.center_freq

    def set_sdr_samplingrate(self, samplingrate=2.4e6):
        """Defines the sampling rate.

        Keyword arguments:
        :param samplingrate -- samplerate [Samples/s] (default 2.4e6)
        """
        self.__sdr.sample_rate = samplingrate

    def get_sdr_samplingrate(self,bprintout=False):
        """Returns sample rate assigned to object and gives default tuner value.
        range is between 1.0 and 3.2 MHz
        """
        if bprintout:
            print ('Default sample rate: 2.4MHz')
            print ('Current sample rate: ' + str(self.__sdr.sample_rate / 1e6) + 'MHz')
        return self.__sdr.sample_rate

    def set_freqtx(self, freqtx_list):
        """ Defines the frequencies on which the beacons transmit power

        :param freqtx_list:  [Hz] list with frequencies on which the beacons transmit power
        """
        self.__freqtx = freqtx_list
        self.__numoftx = len(freqtx_list)

    def get_freqtx(self):
        """

        :return: list of transceiver frequencies
        """
        return self.__freqtx

    def get_numoftx(self):
        """

        :return: number of transceiver
        """
        return self.__numoftx


    def set_txpos(self, txpos):
        """ Set transceiver positions

        :param txpos:
        :return:
        """
        self.__txpos = txpos

    def get_txpos(self):
        """ Get transceiver positions

        :return: [mm] array with positions of all beacons
        """
        return self.__txpos

    """

    Section with general data processing methods

    """

    def get_power_density_spectrum(self):
        """
        gets the iq-samples and calculates the power density.
        It sorts the freq and pden vector such that the frequencies are increasing.
        Moreover the center frequency is added to freq such that freq contains the absolute frequencies for the
        corresponding power density

        :return: freq_sorted, pxx_density_sorted
        """
        samples = self.get_sdr_iq_sample()

        # FFT
        freq, pxx_den = signal.periodogram(samples, fs=self.get_sdr_samplingrate(), nfft=1024)

        # sort the data to get increasing frequencies
        freq_sorted = np.concatenate((freq[len(freq) / 2:], freq[:len(freq) / 2]), axis=0)
        pxx_density_sorted = np.concatenate((pxx_den[len(pxx_den) / 2:], pxx_den[:len(pxx_den) / 2]), axis=0)

        freq_sorted = freq_sorted + self.get_sdr_centerfreq()  # add center frequency to get absolute frequency values

        return freq_sorted, pxx_density_sorted

    def get_rss_peaks_at_freqtx(self, freqtx):
        """
        find maximum rss peaks in spectrum
        :param freqtx: frequency which max power density is looked for
        :return: freq_peaks, rss_peaks
        """

        freqspan = self.get_freqspan()

        freq_spectrum, pxx_den = self.get_power_density_spectrum()

        freq_peaks = []
        rss_peaks = []

        # loop for all tx-frequencies
        for ifreq in range(self.__numoftx):
            startindex = 0
            endindex = len(freq_spectrum)
            i = 0
            # find start index of frequency vector
            while i < len(freq_spectrum):
                if freq_spectrum[i] >= freqtx[ifreq] - freqspan / 2:
                    startindex = i
                    break
                i += 1
            # find end index of frequency vector
            while i < len(freq_spectrum):
                if freq_spectrum[i] >= freqtx[ifreq] + freqspan / 2:
                    endindex = i
                    break
                i += 1

            pxx_den = np.array(pxx_den)

            # find index of the highest power density
            maxind = np.where(pxx_den == max(pxx_den[startindex:endindex]))

            # workaround to avoid that two max values are used in the further code
            maxind = maxind[0]  # converts array to list
            maxind = maxind[0]  # takes first list element

            rss_peaks.append(10 * np.log10(pxx_den[maxind]))
            freq_peaks.append(freq_spectrum[maxind])

        return freq_peaks, rss_peaks

    def get_rss_peaks(self):
        """
        for external use i.e. estimator methods
        finds maximus rss peaks in power spectrum by calling method *get_max_rss_peaks_at_freqtx
        :return: freq_peaks, rss_peaks
        """
        freqtx = self.get_freqtx()
        freq_peaks, rss_peaks = self.get_rss_peaks_at_freqtx(freqtx)

        return freq_peaks, rss_peaks

    def take_measurement(self, meastime):
        """ Takes measurements over defined period of time

        :param meastime: [s] time for taking measurements

        :return: np array of rss-peaks at freqtx
        """
        print ('... measuring for ' + str(meastime) + 's ...')
        elapsed_time = 0.0
        dataseq = []

        while elapsed_time < meastime:
            start_calctime = t.time()
            freq_den_max, pxx_den_max = self.get_rss_peaks_at_freqtx(self.get_freqtx())
            # print('get_freqtx= ' + str(self.get_freqtx()))
            # print('freq_found = ' + str(freq_den_max))

            dataseq.append(pxx_den_max)

            calc_time = t.time() - start_calctime
            elapsed_time = elapsed_time + calc_time
            t.sleep(0.001)

        dataseq_mat = np.asarray(dataseq)
        return dataseq_mat

    def manual_calibration_for_one_tx(self, measdata_filename, meastime=5, b_plotting=True):
        """
        :return:
        """

        print('Manual calibration started!')
        # read data from waypoint file
        freqtx = self.__freqtx
        #wplist_filename = hc_tools.select_file()
        #wp_data_mat, x0, xn, grid_dxdy, timemeas = rf_tools.read_data_from_wp_list_file(wplist_filename)
        meas_mean = []
        meas_var = []
        dist_list = []

        #measdata_filename = hc_tools.save_as_dialog('Save measurement data as...')
        with open(measdata_filename, 'w') as measfile:
            meas_counter = 0
            b_new_measurement = True
            # loop over all way-points
            while b_new_measurement:

                input_dist = raw_input('Next measurement distance [mm]? (type >end< to exit)')

                if input_dist == 'end':
                    break

                meas_counter = meas_counter + 1
                numwp = meas_counter
                meas_point = [input_dist, 0]
                dist_list.append(input_dist)

                print('Measuring at Way-Point #' + str(numwp) + ' at distance ' + str(meas_point[0] + 'mm'))

                dataseq = self.take_measurement(meastime)

                [nummeas, numtx] = np.shape(dataseq)

                # way point data - structure 'wp_x, wp_y, num_wp, num_tx, num_meas'
                str_base_data = str(meas_point[0]) + ', ' + str(meas_point[1]) + ', ' +\
                                str(numwp) + ', ' + str(numtx) + ', ' + str(nummeas) + ', '
                # freq data
                str_freqs = ', '.join(map(str, freqtx)) + ', '

                # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
                # print('data ' + str(dataseq))
                str_rss = ''
                for i in range(numtx):
                    str_rss = str_rss + ', '.join(map(str, dataseq[:, i])) + ', '

                print('Measurements taken: ' + str(nummeas) + ' at sample-size ' + str(self.__samplesize))

                measfile.write(str_base_data + str_freqs + str_rss + '\n')

                meas_mean.append(np.mean(dataseq, axis=0))
                meas_var.append(np.var(dataseq, axis=0))
                print('input_dist = ' + input_dist)
                if b_plotting:
                    self.cal_plot(dist_list, meas_mean, meas_var)

            measfile.close()
            print('Measurement file ' + measdata_filename + ' closed :-)')
        return True

    """

    Plotting methods

    """

    def plot_power_spectrum_density(self):
        """Get Power Spectral Density Live Plot."""
        center_freq = self.get_sdr_centerfreq()

        plt.ion()      # turn interactive mode on
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # init x and y -> y is update with each sample
        x = np.linspace(center_freq-1024e3, center_freq+1022e3, 1024)  # 1024 as the fft has 1024frequency-steps
        y = x
        line1, = ax.plot(x, y, 'b-')

        """ setup plot properties """

        plt.axis([center_freq - 1.1e6, center_freq + 1.1e6, -120, 0])
        xlabels = np.linspace((center_freq-1.0e6)/1e6,
                              (center_freq+1.0e6)/1e6, 21)
        plt.xticks(np.linspace(min(x), max(x), 21), xlabels, rotation='vertical')

        plt.grid()
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [dB]')
        drawing = True
        line1.set_xdata(x)
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                freq, pxx_den = self.get_power_density_spectrum()
                line1.set_ydata(10*np.log10(pxx_den))

                # @todo annotations on the frequency peaks
                # if known_freqtx > 0:
                #    #freq_den_max, pdb_den_max = self.get_max_rss_in_freqspan(known_freqtx, freqspan)
                #    plt.annotate(r'$this is an annotation',
                #                 xy=(433e6, -80), xycoords='data',
                #                 xytext=(+10, +30), textcoords='offset points', fontsize=16,
                #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                fig.canvas.draw()
                plt.pause(0.01)
            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

    def plot_txrss_live(self, numofplottedsamples=250):
        """ Live plot for the measured rss from each tx

        :param numofplottedsamples: number of displayed samples (default= 250)
        :return: 0
        """
        numoftx = self.__numoftx

        if numoftx > 7:
            print('Number of tracked tx needs to be <=7!')  # see length of colorvec
            print('Terminate method!')
            return True

        rss = np.zeros((numoftx, 1))
        temp = np.zeros((numoftx, 1))

        plt.ion()  # turn interactive mode on
        colorvec = ['b', 'r', 'g', 'm', 'c', 'k', 'y']  # all colors which can be used in the plot

        cnt = 0
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                cnt += 1
                # find maximum power peaks in spectrum
                freq_found, pxx_den_max = self.get_rss_peaks()

                for i in range(numoftx):
                    temp[i, 0] = pxx_den_max[i]
                rss = np.append(rss, temp, axis=1)
                # plot data for all tx
                plt.clf()
                firstdata = 1  # set max number of plotted points per tx
                if cnt > numofplottedsamples:
                    firstdata = cnt - numofplottedsamples

                for i in range(numoftx):
                    plt.plot(rss[i, firstdata:-1], str(colorvec[i])+'.-',
                             label="Freq = " + str(round(freq_found[i] / 1e6, 2)) + ' MHz' + '@ ' + str(round(rss[i, -1], 2)) + 'dBm')
                plt.ylim(-120, 10)
                plt.ylabel('RSS [dB]')
                plt.grid()
                plt.legend(loc='upper right')
                plt.pause(0.001)

            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

    def get_performance(self, testfreqtx=434e6, samplingrate=2.4e6):
        """Measure performance for given sampling rates.

        Keyword arguments:
        :param testfreqtx -- single frequency for which the performence is determined
        :param samplingrate -- sampling rate of sdr [Ms/s] (default 2.4e6)
        """
        print('Performance test started!')
        freqtx = [testfreqtx]
        self.set_sdr_centerfreq(np.mean(freqtx))
        self.set_sdr_samplingrate(samplingrate)
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
                cnt += 1
                start_calctime = t.time()
                # use matplotlib to estimate the PSD and save the max power
                self.set_samplesize(i)
                freqmax, pxx_max = self.get_rss_peaks_at_freqtx(freqtx)
                powerstack.append(pxx_max)
                t.sleep(0.005)
                calctime = t.time() - start_calctime
                timestack.append(calctime)
                elapsed_time = elapsed_time + calctime
            calctime = np.mean(timestack)
            VAR.append(np.var(powerstack))
            MEAN.append(np.mean(powerstack))
            UPDATE.append(calctime)
            total_time += elapsed_time
            print (str(measurements) + ' measurements for batch-size ' + str(self.set_samplesize()) +
                   ' * 1024 finished after ' + str(elapsed_time) + 's. => ' + str(measurements/elapsed_time) + 'Hz')
        print('')
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

    def lambertloc(self, rss, numtx=0):
        """Inverse function of the RSM. Returns estimated range in [mm].

        Keyword arguments:
        :param rss -- received power values [dB]
        :param numtx  -- number of the tx which rss is processed. Required to use the corresponding alpha and gamma-values.
        """
        z = 20 / (np.log(10) * self.__alpha[numtx]) * lambertw(
            np.log(10) * self.__alpha[numtx] / 20 * np.exp(-np.log(10) / 20 * (rss + self.__gamma[numtx])))
        return z.real  # [mm]



