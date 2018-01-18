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

    #__metaclass__ = ABCMeta

    def __init__(self, center_freq, freqspan=2e4):
        """
        init-method
        :param center_freq: [Hz] Defines the center frequency where to listen (between 27MHz and 1.7GHz)
        :param freqspan: [Hz] span within the the algorithm is looking for amplitude peaks
        """
        # connect to sdr
        self.__sdr = RtlSdr()
        self.__sdr.gain = 1
        self.__sdr.sample_rate = 2.048e6

        self.__centerfreq = center_freq
        self.set_sdr_centerfreq(self.__centerfreq)

        self.__freqspan = freqspan
        self.__samplesize = 0
        self.set_samplesize(32)

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

    def get_rss_peaks(self):
        """
        for external use i.e. estimator methods
        finds maximus rss peaks in power spectrum by calling method *get_max_rss_peaks_at_freqtx
        :return: freq_peaks, rss_peaks
        """
        freqtx = self.get_freqtx()
        freq_peaks, rss_peaks = self.get_rss_peaks_at_freqtx(freqtx)

        return freq_peaks, rss_peaks

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
        for ifreq in range(len(freqtx)):
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

    def take_measurement(self, meastime):
        """ Takes measurements over defined persiod of time

        :param meastime: [s] time for taking measurements

        :return: np array of rss-peaks at freqtx
        """
        print ('... measuring for ' + str(meastime) + 's ...')
        elapsed_time = 0.0
        dataseq = []

        while elapsed_time < meastime:
            start_calctime = t.time()
            freq_den_max, pxx_den_max = self.get_rss_peaks_at_freqtx(self.get_freqtx())

            dataseq.append(pxx_den_max)

            calc_time = t.time() - start_calctime
            elapsed_time = elapsed_time + calc_time
            t.sleep(0.001)

        dataseq_mat = np.asarray(dataseq)
        return dataseq_mat

    def manual_calibration_for_one_tx(self, measdata_filename, meastime = 5, b_plotting=True):
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

    def cal_plot(self, r_dist, rss_mean, rss_var):


#        rdist = np.array(rdist_temp[itx, :], dtype=float)
#        rss_mean = np.array(rss_mean, dtype=float)
#        rss_var = np.array(rss_var, dtype=float)
        plt.figure(1)


        #r_dist = np.array(rdist_temp, dtype=float)
        #rss_mean = np.array(rss_mean, dtype=float)
        #rss_var = np.array(rss_var, dtype=float)
        print(np.shape(r_dist))
        print(np.shape(rss_mean))
        print(np.shape(rss_mean))
        print(r_dist)
        print(rss_mean)
        plt.plot(r_dist, rss_mean)
        #plt.errorbar(r_dist, rss_mean, yerr=rss_var,
        #        fmt='ro', markersize='1', ecolor='g', label='Original Data')


        #rdata = np.linspace(np.min(rdist), np.max(rdist), num=1000)
        #ax.plot(rdata, rsm_model(rdata, alpha[itx], gamma[itx]), label='Fitted Curve')
        plt.legend(loc='upper right')
        plt.grid()
        plt.ylim([-110, -10])
        plt.xlabel('Distance [mm]')
        plt.ylabel('RSS [dB]')
        plt.title('RSM for TX# 1')
        plt.show()

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

                ## @todo annotations on the frequency peaks
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

        rdist = np.zeros((numoftx, 1))
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
                rdist = np.append(rdist, temp, axis=1)

                # plot data for all tx
                plt.clf()
                firstdata = 1  # set max number of plotted points per tx
                if cnt > numofplottedsamples:
                    firstdata = cnt - numofplottedsamples

                for i in range(numoftx):
                    plt.plot(rdist[i, firstdata:-1], str(colorvec[i])+'.-',
                             label="Freq = " + str(round(freq_found[i] / 1e6, 2)) + ' MHz' + '@ ' + str(round(rdist[i, -1],2)) + 'dBm')
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
        """Measure performance at certain sizes and sampling rates.

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


    def set_txpos(self, txpos):
        """

        :param txpos:
        :return:
        """
        self.__txpos = txpos  # @todo: implement a dimension check

    def get_txpos(self):
        """

        :return: [mm] array with positions of all beacons
        """
        return self.__txpos

    def calibrate(self, numtx=0, time=5.0):
        """Adjust RSM in line with measurement.
        :param numtx - number of the tx which needs to be calibrated
        :param time - time for calibration measurement in [s]
        """

        dist_ref = raw_input('Please enter distance '
                             'from transmitter to receiver [mm]: ')

        def rsm_func(ref, gamma_diff_cal):
            """RSM structure with correction param gamma_diff_cal."""
            return -20 * np.log10(ref[0]) - ref[1] * ref[0] - ref[2] - gamma_diff_cal  # ... -gamma -gamma_diff

        elapsed_time = 0.0
        powerstack = []

        # take first sample after boot dvb-t-dongle and delete it
        firstsample = self.get_sdr_iq_sample()
        del firstsample

        # get measurements
        print (' ... measuring ' + str(time) + 's ...')
        while elapsed_time < time:
            start_calctime = t.time()
            freq_den_max, pxx_den_max = self.get_rss_peaks()
            powerstack.append(pxx_den_max[numtx])
            calc_time = t.time() - start_calctime
            elapsed_time = elapsed_time + calc_time
            t.sleep(0.01)
        print ('done\n')
        t.sleep(0.5)

        print (' ... evaluating ...')
        powerstack = np.array(powerstack)
        p_mean = []
        p_mean.append(np.mean(powerstack))  # powerstack is a vector of all rss-peaks

        print ('Variance [dB]:')
        print (np.var(powerstack))

        print ('Calibration reference frequency: ' + str(self.__freqtx[numtx] / 1e6) + ' MHz')
        dist_ref = np.array(dist_ref, dtype=float)
        p_ref = p_mean

        # curve fit with calibrated value -
        # dist_ref, alpha, gamma are fixed -> gamma_diff_opt is the change in gamma by new meas
        gamma_diff_opt, pcov = curve_fit(rsm_func, [dist_ref, self.__alpha[numtx], self.__gamma[numtx]], p_ref)
        del pcov
        print ('gamma old: ' + str(self.__gamma[numtx]))
        self.__gamma[numtx] = self.__gamma[numtx] + gamma_diff_opt[0]  # update gamma with calibration
        print ('gamma_diff: ' + str(gamma_diff_opt[0]))
        print ('gamma new: ' + str(self.__gamma[numtx]))

    def get_caldata(self, numtx=0):
        """Returns the calibrated RSM params."""
        return self.__alpha[numtx], self.__gamma[numtx]





    def map_path_ekf(self, x0, h_func_select, bplot=True, blog=False, bprintdata=False):
        """ map/track the position of the mobile node using an EKF

        Keyword arguments:
        :param x0 -- initial estimate of the mobile node position
        :param h_func_select:
        :param bplot -- Activate/Deactivate liveplotting the data (True/False)
        :param blog -- activate data logging to file (default: False)
        :param bprintdata - activate data print to console(default: False)
        """

        # measurement function
        def h_dist(x, txpos, numtx):
            tx_pos = txpos[numtx]  # position of the transceiver
            # r = sqrt((x-x_tx)^2+(y-y_tx)^2)
            y_dist = np.sqrt((x[0]-tx_pos[0])**2+(x[1]-tx_pos[1])**2)
            return y_dist

        # jacobian of the measurement function
        def h_dist_jacobian(x_est, txpos, numtx):
            tx_pos = txpos[numtx]  # position of the transceiver
            factor = 0.5/np.sqrt((x_est[0]-tx_pos[0])**2+(x_est[1]-tx_pos[1])**2)
            h_dist_jac = np.array([factor*2*(x_est[0]-tx_pos[0]), factor*2*(x_est[1]-tx_pos[1])])  # = [dh/dx1, dh/dx2]
            return h_dist_jac

        def h_rss(x, tx_param, numtx):
            tx_param_temp = tx_param[numtx]
            tx_pos = tx_param_temp[0]  # position of the transceiver
            alpha = tx_param_temp[1]
            gamma = tx_param_temp[2]

            # r = sqrt((x - x_tx) ^ 2 + (y - y_tx) ^ 2)
            r_dist = np.sqrt((x[0] - tx_pos[0])**2 + (x[1] - tx_pos[1])**2)
            y_rss = -20 * np.log10(r_dist) - alpha * r_dist - gamma

            return y_rss

        def h_rss_jacobian(x_est, tx_param, numtx):
            tx_param_temp = tx_param[numtx]
            tx_pos = tx_param_temp[0]  # position of the transceiver
            alpha = tx_param_temp[1]
            # gamma = tx_param_temp[2]  # not used here

            R_dist = np.sqrt((x_est[0] - tx_pos[0])**2 + (x_est[1] - tx_pos[1])**2)

            # dh / dx1
            h_rss_jac_x = -20 * (x_est[0] - tx_pos[0]) / (np.log(10) * R_dist**2) - alpha * (x_est[0] - tx_pos[0]) / R_dist
            # dh / dx2
            h_rss_jac_y = -20 * (x_est[1] - tx_pos[1]) / (np.log(10) * R_dist**2) - alpha * (x_est[1] - tx_pos[1]) / R_dist

            h_rss_jac = np.array([[h_rss_jac_x], [h_rss_jac_y]])

            return h_rss_jac

        if h_func_select == 'h_dist':
            h = h_dist
            h_jacobian = h_dist_jacobian
        elif h_func_select == 'h_rss':
            h = h_rss
            h_jacobian = h_rss_jacobian
        else:
            print ('You need to select to a measurement function "h" like "h_rss" or "h_dist"!')
            print ('exit...')
            return True

        txpos = self.__txpos
        """ setup figure """
        if bplot:
            plt.ion()
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(111)

            x_min = -500.0
            x_max = 3000.0
            y_min = -500.0
            y_max = 2000.0
            plt.axis([x_min, x_max, y_min, y_max])

            plt.grid()
            plt.xlabel('x-Axis [mm]')
            plt.ylabel('y-Axis [mm]')

            for i in range(self.__numoftx):
                txpos_single = txpos[i]
                ax.plot(txpos_single[0], txpos_single[1], 'ro')

            # init measurement circles and add them to the plot
            circle_meas = []
            circle_meas_est = []
            for i in range(self.__numoftx):
                txpos_single = txpos[i]
                circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='r', fill=False))
                ax.add_artist(circle_meas[i])
                circle_meas_est.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='g', fill=False))
                ax.add_artist(circle_meas_est[i])
        """ initialize tracking setup """
        print(str(self.__txalpha))
        print(str(self.__txgamma))
        print(str(self.__txpos))
        tx_param = []
        for itx in range(self.__numoftx):
            tx_param.append([txpos[itx],self.__txalpha[itx], self.__txgamma[itx]])
        print(str(np.asarray(tx_param)))
        """ initialize EKF """
        # standard deviations
        sig_x1 = 500
        sig_x2 = 500
        p_mat = np.array(np.diag([sig_x1 ** 2, sig_x2 ** 2]))

        # process noise
        sig_w1 = 50
        sig_w2 = 50
        q_mat = np.array(np.diag([sig_w1 ** 2, sig_w2 ** 2]))

        # measurement noise
        sig_r = 10
        r_mat = sig_r ** 2

        # initial values and system dynamic (=eye)
        x_log = np.array([[x0[0]], [x0[1]]])
        x_est = x_log

        i_mat = np.eye(2)

        z_meas = np.zeros(self.__numoftx)
        y_est = np.zeros(self.__numoftx)

        """ Start EKF-loop"""
        tracking = True
        while tracking:
            try:
                # iterate through all tx-rss-values
                freq_den_max, rss = self.get_rss_peaks()
                x_est[:, 0] = x_log[:, -1]

                z_meas = rss
                for itx in range(self.__numoftx):

                    """ prediction """
                    x_est = x_est  #+ np.random.randn(2, 1) * 1  # = I * x_est

                    p_mat_est = i_mat.dot(p_mat.dot(i_mat)) + q_mat

                    # print('x_est: ' + str(x_est))

                    """ update """
                    # estimate measurement from x_est
                    y_est[itx] = h(x_est, tx_param, itx)
                    y_tild = z_meas[itx] - y_est[itx]

                    # calc K-gain
                    h_jac_mat = h_jacobian(x_est[:, 0], tx_param, itx)
                    s_mat = np.dot(h_jac_mat.transpose(), np.dot(p_mat, h_jac_mat)) + r_mat  # = H^t * P * H + R
                    k_mat = np.dot(p_mat, h_jac_mat / s_mat)  # 1/s_scal since s_mat is dim = 1x1

                    x_est = x_est + k_mat * y_tild  # = x_est + k * y_tild
                    p_mat = (i_mat - np.dot(k_mat, h_jac_mat.transpose())) * p_mat_est  # = (I-KH)*P

                x_log = np.append(x_log, x_est, axis=1)
                print(str(z_meas-y_est))
                """ update figure / plot after all measurements are processed """
                if bplot:
                    # add new x_est to plot
                    ax.plot(x_est[0, -1], x_est[1, -1], 'bo')
                    # update measurement circles around tx-nodes

                    #for i in range(self.__numoftx):
                    #    circle_meas[i].set_radius(z_meas[i])
                    #    circle_meas_est[i].set_radius(y_est[i])

                    # update figure 1
                    fig1.canvas.draw()
                    plt.pause(0.001)  # pause to allow for keyboard inputs

                if bprintdata:
                    print(str(x_est) + ', ' + str(p_mat))  # print data to console

            except KeyboardInterrupt:
                print ('Localization interrupted by user')
                tracking = False

        if blog:
            print ('Logging mode enabled')
            print ('TODO: implement code to write data to file')
            # write_to_file(x_log, 'I am a log file')

        if bplot:
            fig2 = plt.figure(2)
            ax21 = fig2.add_subplot(211)
            ax21.grid()
            ax21.set_ylabel('x-position [mm]')
            ax21.plot(x_log[0, :], 'b-')

            ax22 = fig2.add_subplot(212)
            ax22.grid()
            ax22.set_ylabel('y-position [mm]')
            ax22.plot(x_log[1, :], 'b-')

            fig2.canvas.draw()

            raw_input('Press Enter to close the figure and terminate the method!')

        return x_est

    def lambertloc(self, rss, numtx=0):
        """Inverse function of the RSM. Returns estimated range in [cm].

        Keyword arguments:
        :param rss -- received power values [dB]
        :param numtx  -- number of the tx which rss is processed. Required to use the corresponding alpha and gamma-values.
        """
        z = 20 / (np.log(10) * self.__alpha[numtx]) * lambertw(
            np.log(10) * self.__alpha[numtx] / 20 * np.exp(-np.log(10) / 20 * (rss + self.__gamma[numtx])))
        return z.real  # [mm]

    def plot_txdist_live(self,freqspan=2e4, numofplottedsamples=250):
        """ Live plot for the measured distances from each tx using rss

        :param freqspan: width of the frequencyspan around the tracked frq
        :param numofplottedsamples: number of displayed samples (default= 250)
        :return: 0
        """

        freq = 0  # for later access through a parameter. In this version all freqtx are tracked
        if freq == 0:
            freq = self.__freqtx

        numoftx = self.__numoftx
        if numoftx > 7:
            print('Number of tracked tx needs to be <=7!') # see length of colorvec
            print('Terminate method!')
            return True

        rdist = np.zeros((numoftx, 1))
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
                freq_found, pxx_den_max = self.get_max_rss_in_freqspan(freq, freqspan)

                for i in range(numoftx):
                    temp[i, 0] = self.lambertloc(pxx_den_max[i], i)
                rdist = np.append(rdist, temp, axis=1)

                # plot data for all tx
                plt.clf()
                firstdata = 1  # set max number of plotted points per tx
                if cnt > numofplottedsamples:
                    firstdata = cnt - numofplottedsamples

                for i in range(numoftx):
                    plt.plot(rdist[i, firstdata:-1], str(colorvec[i])+'.-',
                             label="Freq = " + str(freq_found[i] / 1e6) + ' MHz')
                plt.ylim(-100, 3000)
                plt.ylabel('R [mm]')
                plt.grid()
                plt.legend(loc='upper right')
                plt.pause(0.001)

            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

    def rfear_type(self):
        """Return a string representing the type of RfEar this is."""
        print ('RfEar,')
        print ('Number of TX: ' + str(self.__numoftx))
        print ('Alpha: ' + str(self.__alpha) + ', gamma: ' + str(self.__gamma))
        print ('Tuned to:' + str(self.get_freq()) + ' MHz,')
        self.get_srate()
        print ('Reads ' + str(self.get_size()) + '*1024 8-bit I/Q-samples from SDR device.')

