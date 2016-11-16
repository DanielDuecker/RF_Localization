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
        :param *args -- list of frequencies (must be in a range of __sdr.sample_rate)
        """
        self.__freq = []
        self.set_freq(self, *args)
        self.__size = 0
        self.set_size(256)

    def set_size(self, size):
        """Set number of samples to be read by sdr [*1024].
        Keyword arguments:
        :param size -- size of the samples be read by sdr [*1024]
        """
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
        :param *args -- single frequency or array of frequencies (default none)
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
        :param srate -- samplerate [Samples/s] (default 2.4e6)
        """
        sdr.sample_rate = srate

    def get_srate(self):
        """Returns sample rate assigned to object and gives default tuner value.
        range is between 1.0 and 3.2 MHz
        """
        print ('Default sample rate: 2.4MHz')
        print ('Current sample rate: ' + str(sdr.sample_rate / 1e6) + 'MHz')

    def print_pxx_density(self):
        """ method to print the sorted power density values to console

        :return:
        """
        printing = True
        while printing:
            try:
                freq, pxx_den = self.get_absfreq_pden_sorted()
                print(str(pxx_den))
                plt.pause(0.001)

            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                printing = False
        return True

    def wp_generator(self, wp_filename='wplist.txt', x0=[0, 0], xn=[1000, 1000], steps=[11, 11], timemeas=10.0):
        """
        :param wp_filename:
        :param x0: [x0,y0] - start position of the grid
        :param xn: [xn,yn] - end position of the grid
        :param steps: [numX, numY] - step size
        :param timestep: - time [s] to wait at each position for measurements
        :return: wp_mat [x, y, t]
        """
        startx = x0[0]
        endx = xn[0]
        stepx = steps[0]

        starty = x0[1]
        endy = xn[1]
        stepy = steps[1]

        xpos = np.linspace(startx, endx, stepx)
        ypos = np.linspace(starty, endy, stepy)

        wp_matx, wp_maty = np.meshgrid(xpos, ypos)
        wp_vecx = np.reshape(wp_matx, (len(xpos)*len(ypos), 1))
        wp_vecy = np.reshape(wp_maty, (len(ypos)*len(xpos), 1))
        wp_time = np.ones((len(xpos)*len(ypos), 1)) * timemeas

        wp_mat = np.append(wp_vecx, wp_vecy, axis=1)
        wp_mat = np.append(wp_mat, wp_time, axis=1)

        plt.figure()
        plt.plot(wp_mat[:, 0], wp_mat[:, 1], '.-')
        plt.show()

        with open(wp_filename, 'a') as wpfile:
            # wpfile.write(t.ctime() + '\n')
            # wpfile.write('some describtion' + '\n')
            for i in range(wp_mat.shape[0]):
                wpfile.write(str(i) + ', ' + str(wp_mat[i, 0]) + ', ' + str(wp_mat[i, 1]) + ', ' + str(wp_mat[i, 2]) + '\n')
            wpfile.close()

        return wp_filename  # file output [line#, x, y, time]

    def write_sample_sequence_to_file(self, ofile, wp, time, numsample, sampleseq):
        """

        :param ofile:
        :param wp:
        :param time:
        :param numsample:
        :param sampleseq:
        :return:
        """
        strrow = str(wp[0]) + ', ' + str(wp[1]) + ', ' + str(time) + ', ' + str(numsample) + ', ' + str(sampleseq)
        ofile.write(strrow + '\n')

        return True

    def measure_at_waypoint(self, wplist_filename, measfilename):
        """

        :param wplist_filename:
        :return:
        """
        measfile = open(measfilename, 'a')

        with open(wplist_filename, 'r') as wpfile:
            measfile.write(t.ctime() + '\n')
            measfile.write('some describtion' + '\n')
            measfile.write('\n')

            # loop through wp-list
            for line in wpfile:
                print line
                if 'str' in line:
                    break
                measfile.write(line)
                elapsed_time = 0.0
                sampleseq = []

            time = 3.14  # debug value

            # get measurements
            print (' ... measuring ' + str(time) + 's ')  # 'at wp #x/totalwp @ pos (x,y)
            """
            while elapsed_time < time:

                start_calctime = t.time()

                # freq_den_max, pxx_den_max = self.get_max_rss_in_freqspan(self.__freqtx, self.__freqspan)

                sampleseq.append(self.get_iq())
                # powerstack.append(pxx_den_max[numtx])

                calc_time = t.time() - start_calctime
                elapsed_time = elapsed_time + calc_time
                t.sleep(0.01)
            numsample = len(sampleseq)
            self.write_sample_sequence_to_file(ofile, wp, meastime, numsample, sampleseq)
            print ('done\n')
            t.sleep(0.5)
            """

    def plot_psd(self):
        """Get Power Spectral Density Live Plot."""

        sdr.center_freq = np.mean(self.__freq)
        plt.ion()      # turn interactive mode on
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # init x and y -> y is update with each sample
        x = np.linspace(sdr.center_freq-1024e3, sdr.center_freq+1022e3, 1024)  # 1024 as the fft has 1024frequency-steps
        y = x
        line1, = ax.plot(x, y, 'b-')
        """ setup plot properties """
        plt.axis([sdr.center_freq - 1.5e6, sdr.center_freq + 1.5e6, -120, 0])
        xlabels = np.linspace((sdr.center_freq-.5*sdr.sample_rate)/1e6,
                              (sdr.center_freq+.5*sdr.sample_rate)/1e6, 5)
        plt.xticks(np.linspace(min(x), max(x), 5), xlabels, rotation='horizontal')
        plt.grid()
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Power [dB]')
        drawing = True
        while drawing:
            try:
                # Busy-wait for keyboard interrupt (Ctrl+C)
                freq, pxx_den = self.get_absfreq_pden_sorted()

                line1.set_ydata(10*np.log10(pxx_den))

                ## @todo annotations on the frequency peaks
                # if known_freqtx > 0:
                #    #freq_den_max, pdb_den_max = self.get_max_rss_in_freqspan(known_freqtx, freqspan)
                #    plt.annotate(r'$this is an annotation',
                #                 xy=(433e6, -80), xycoords='data',
                #                 xytext=(+10, +30), textcoords='offset points', fontsize=16,
                #                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
                fig.canvas.draw()
                plt.pause(0.001)
            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

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

        :return: freqsort, pxx_densort
        """
        samples = self.get_iq()
        freq, pxx_den = signal.periodogram(samples,
                                           fs=sdr.sample_rate, nfft=1024)

        freq_sorted = np.concatenate((freq[len(freq) / 2:], freq[:len(freq) / 2]), axis=1)
        pxx_den_sorted = np.concatenate((pxx_den[len(pxx_den) / 2:], pxx_den[:len(pxx_den) / 2]), axis=1)

        freq_sorted = freq_sorted + sdr.center_freq # add centerfreq to get absolut frequency values

        return freq_sorted, pxx_den_sorted

    def get_max_rss_in_freqspan(self, freqtx, freqspan):
        """
        find maximum rss peaks in spectrum
        :param freqtx: frequency which max power density is looked for
        :param freqspan: width of the frequency span
        :return: frequeny, maxpower
        """

        freq, pxx_den = self.get_absfreq_pden_sorted()

        freq_den_max = []
        pdb_den_max = []

        # loop for all tx-frequencies
        for ifreq in range(len(freqtx)):
            startindex = 0
            endindex = len(freq)
            i = 0
            # find start index of frequency vector
            while i < len(freq):
                if freq[i] >= freqtx[ifreq] - freqspan / 2:
                    startindex = i
                    break
                i += 1
            # find end index of frequency vector
            while i < len(freq):
                if freq[i] >= freqtx[ifreq] + freqspan / 2:
                    endindex = i
                    break
                i += 1

            pxx_den = np.array(pxx_den)

            # find index of the highest power density
            maxind = np.where(pxx_den == max(pxx_den[startindex:endindex]))

            # workaround to avoid that two max values are used in the further code
            maxind = maxind[0]  # converts array to list
            maxind = maxind[0]  # takes first list element

            pdb_den_max.append(10 * np.log10(pxx_den[maxind]))
            freq_den_max.append(freq[maxind])

        return freq_den_max, pdb_den_max

    def plot_txrss_live(self, freq, freqspan=2e4, numofplottedsamples=250):
        """ Live plot for the measured rss from each tx

        :param freqspan: width of the frequencyspan around the tracked frq
        :param numofplottedsamples: number of displayed samples (default= 250)
        :return: 0
        """
        numoftx = len(freq)
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
                    temp[i, 0] = pxx_den_max[i]
                rdist = np.append(rdist, temp, axis=1)

                # plot data for all tx
                plt.clf()
                firstdata = 1  # set max number of plotted points per tx
                if cnt > numofplottedsamples:
                    firstdata = cnt - numofplottedsamples

                for i in range(numoftx):
                    plt.plot(rdist[i, firstdata:-1], str(colorvec[i])+'.-',
                             label="Freq = " + str(freq_found[i] / 1e6) + ' MHz')
                plt.ylim(-120, 10)
                plt.ylabel('RSS [dB]')
                plt.grid()
                plt.legend(loc='upper right')
                plt.pause(0.001)

            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

    @abstractmethod
    def rfear_type(self):
        """Return a string representing the type of rfear this is."""
        pass

    #unused
    def rpi_get_power(self, printing=0, size=256):
        """Routine for Raspberry Pi.

        Keyword arguments:
        :param printing -- visible output on terminal (default  0)
        :param size -- measure for length of fft (default 256*1024)
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

    def measure_rss_var(self, freqtx, freqrange=2e4, time=10.0):
        """
        Interactive method to get PSD data
        at characteristic frequencies.
        :param freqtx: tx-frequency [Hz]
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
        print ('RSS ist measured at freq: ' + str(freqtx[0] / 1e6) +
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
                    freqs, rss = self.get_max_rss_in_freqspan(freqtx, freqrange)
                    del freqs
                    powerstack.append(rss)
                    calc_time = t.time() - start_calctime
                    elapsed_time = elapsed_time + calc_time
                    t.sleep(0.01)
                print ('done\n')
                t.sleep(0.5)
                print (' ... evaluating ...')

                powerstack.pop(0)  # uggly workaround to ignore first element after dvb-t boot up

                modeldata.append(np.mean(powerstack))
                variance.append(np.var(powerstack))

                plt.clf()
                plt.errorbar(range(len(modeldata)), modeldata, yerr=variance, fmt='o', ecolor='g')
                plt.xlabel('# of Evaluations')
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
        def func(dist, alpha, xi):
            """Range Sensor Model (RSM) structure."""
            return -20*np.log10(dist)-alpha*dist-xi
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

    def get_performance(self, bandwidth=2.4e6):
        """Measure performance at certain sizes and sampling rates.

        Keyword arguments:
        :param bandwidth -- sampling rate of sdr [Ms/s] (default 2.4e6)
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
                cnt += 1
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


class LocEar(RfEar):
    """Subclass of Superclass RfEar for 2D dynamic object localization."""
    def __init__(self, alpha, xi, freqtx, freqspan, *args):
        RfEar.__init__(self, *args)
        self.__alpha = alpha
        self.__xi = xi
        self.__freqtx = freqtx
        self.__freqspan = freqspan
        self.__numoftx = len(freqtx)

    def calibrate(self, numtx=0, time=20.0):
        """Adjust RSM in line with measurement.
        :param numtx - number of the tx which needs to be calibrated
        :param time - time for calibration measurement in [s]
        """

        dist_ref = raw_input('Please enter distance '
                             'from transmitter to receiver [cm]: ')

        def rsm_func(ref, xi_diff_cal):
            """RSM structure with correction param xi_diff_cal."""
            return -20 * np.log10(ref[0]) - ref[1] * ref[0] - ref[2] - xi_diff_cal  # ... -xi -xi_diff

        elapsed_time = 0.0
        powerstack = []

        # take first sample after boot dvb-t-dongle and delete it
        firstsample = self.get_size()
        del firstsample

        # get measurements
        print (' ... measuring ' + str(time) + 's ...')
        while elapsed_time < time:
            start_calctime = t.time()
            freq_den_max, pxx_den_max = self.get_max_rss_in_freqspan(self.__freqtx, self.__freqspan)
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

        # curve fit with calibrated value - dist_ref, alpha, xi are fixed -> xi_diff_opt is the change in xi by new meas
        xi_diff_opt, pcov = curve_fit(rsm_func, [dist_ref, self.__alpha[numtx], self.__xi[numtx]], p_ref)
        del pcov
        print ('Xi alt: ' + str(self.__xi[numtx]))
        self.__xi[numtx] = self.__xi[numtx] + xi_diff_opt[0]  # update xi with calibration
        print ('Xi_diff: ' + str(xi_diff_opt[0]))
        print ('Xi neu: ' + str(self.__xi[numtx]))

    def get_caldata(self, numtx=0):
        """Returns the calibrated RSM params."""
        return self.__alpha[numtx], self.__xi[numtx]



    def map_path_ekf(self, x0, txpos, bplot=True, blog=False, bprintdata=False):
        """ map/track the position of the mobile node using an EKF

        Keyword arguments:
        :param x0 -- initial estimate of the mobile node position
        :param txpos -- vector of tx positions [x,y], first tx is origin of coordinate frame [cm]
        :param bplot -- Activate/Deactivate liveplotting the data (True/False)
        :param blog -- activate data logging to file (default: False)
        """

        # measurement function
        def h_meas(x, txposition, numtx):
            tx_pos = txposition[numtx, :]  # position of the transceiver
            # r = sqrt((x-x_tx)^2+(y-y_tx)^2)
            r_dist = np.sqrt((x[0]-tx_pos[0])**2+(x[1]-tx_pos[1])**2)
            return r_dist

        # jacobian of the measurement function
        def h_jacobian(x_est, txpos, numtx):
            tx_pos = txpos[numtx, :]  # position of the transceiver
            factor = 0.5/np.sqrt((x_est[0]-tx_pos[0])**2+(x_est[1]-tx_pos[1])**2)
            h_jac = np.array([factor*2*(x_est[0]-tx_pos[0]), factor*2*(x_est[1]-tx_pos[1])])  # = [dh/dx1, dh/dx2]
            return h_jac

        """ setup figure """
        if bplot:
            plt.ion()
            fig1 = plt.figure(1)
            ax = fig1.add_subplot(111)

            x_min = -50.0
            x_max = 150.0
            y_min = -50.0
            y_max = 200.0
            plt.axis([x_min, x_max, y_min, y_max])

            plt.grid()
            plt.xlabel('x-Axis [cm]')
            plt.ylabel('y-Axis [cm]')

            for i in range(self.__numoftx):
                ax.plot(txpos[i, 0], txpos[i, 1], 'ro')

            # init measurement circles and add them to the plot
            circle_meas = []
            circle_meas_est = []
            for i in range(self.__numoftx):
                circle_meas.append(plt.Circle((txpos[i, 0], txpos[i, 1]), 0.01, color='r', fill=False))
                ax.add_artist(circle_meas[i])
                circle_meas_est.append(plt.Circle((txpos[i, 0], txpos[i, 1]), 0.01, color='g', fill=False))
                ax.add_artist(circle_meas_est[i])

        """ initialize EKF """
        # standard deviations
        sig_x1 = 50
        sig_x2 = 50
        p_mat = np.array(np.diag([sig_x1 ** 2, sig_x2 ** 2]))

        # process noise
        sig_w1 = 2
        sig_w2 = 2
        q_mat = np.array(np.diag([sig_w1 ** 2, sig_w2 ** 2]))

        # measurement noise
        sig_r = 1
        r_mat = sig_r ** 2

        # initial values and system dynamic (=eye)
        x_log = np.array([[x0[0]], [x0[1]]])
        x_est = x_log

        i_mat = np.eye(2)

        z_meas = [0, 0, 0]
        y_est = [0, 0, 0]

        """ Start EKF-loop"""
        tracking = True
        while tracking:
            try:
                # iterate through all tx-rss-values
                freq_den_max, rss = self.get_max_rss_in_freqspan(self.__freqtx, self.__freqspan)
                x_est[:, 0] = x_log[:, -1]

                for i in range(self.__numoftx):

                    """ prediction """
                    x_est[:, 0] = x_est[:, 0] + np.random.randn(1, 2) * 1  # = I * x_est

                    p_mat_est = i_mat.dot(p_mat.dot(i_mat)) + q_mat

                    # print('x_est: ' + str(x_est))

                    """ update """
                    # get new measurement / get distance from rss-measurement
                    z_meas[i] = self.lambertloc(rss[i], i)
                    # estimate measurement from x_est
                    y_est[i] = h_meas(x_est[:, 0], txpos, i)
                    y_tild = z_meas[i] - y_est[i]

                    # calc K-gain
                    h_jac_mat = h_jacobian(x_est[:, 0], txpos, i)
                    s_mat = np.dot(h_jac_mat.transpose(), np.dot(p_mat, h_jac_mat)) + r_mat  # = H^t * P * H + R
                    k_mat = np.dot(p_mat, h_jac_mat.transpose() / s_mat)  # 1/s_scal since s_mat is dim = 1x1

                    x_est[:, 0] = x_est[:, 0] + k_mat * y_tild  # = x_est + k * y_tild
                    p_mat = (i_mat - k_mat.dot(h_jac_mat)) * p_mat_est  # = (I-KH)*P

                x_log = np.append(x_log, x_est, axis=1)

                """ update figure / plot after all measurements are processed """
                if bplot:
                    # add new x_est to plot
                    ax.plot(x_est[0, -1], x_est[1, -1], 'bo')
                    # update measurement circles around tx-nodes
                    for i in range(self.__numoftx):
                        circle_meas[i].set_radius(z_meas[i])
                        circle_meas_est[i].set_radius(y_est[i])
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
            ax21.set_ylabel('x-position [cm]')
            ax21.plot(x_log[0, :], 'b-')

            ax22 = fig2.add_subplot(212)
            ax22.grid()
            ax22.set_ylabel('y-position [cm]')
            ax22.plot(x_log[1, :], 'b-')

            fig2.canvas.draw()

            raw_input('Press Enter to close the figure and terminate the method!')

        return x_est

    def lambertloc(self, rss, numtx=0):
        """Inverse function of the RSM. Returns estimated range in [cm].

        Keyword arguments:
        :param rss -- received power values [dB]
        :param numtx  -- number of the tx which rss is processed. Required to use the corresponding alpha and xi-values.
        """
        z = 20 / (np.log(10) * self.__alpha[numtx]) * lambertw(
            np.log(10) * self.__alpha[numtx] / 20 * np.exp(-np.log(10) / 20 * (rss + self.__xi[numtx])))
        return z.real

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
                plt.ylim(-10, 300)
                plt.ylabel('R [cm]')
                plt.grid()
                plt.legend(loc='upper right')
                plt.pause(0.001)

            except KeyboardInterrupt:
                print ('Liveplot interrupted by user')
                drawing = False
        return True

    def map_path(self, dist_tx):
        """Maps estimated location in 1D or 2D respectively.

        Keyword arguments:
        :param dist_tx -- distance between the transmitting stations [cm] (default 55.0)
        """
        sdr.center_freq = np.mean(self.get_freq())
        x_min = -10.0
        x_max = dist_tx+10.0
        y_min = -100.0
        y_max = 100.0
        plt.axis([x_min, x_max, y_min, y_max])
        plt.ion()
        plt.grid()
        plt.xlabel('x-Axis [cm]')
        plt.ylabel('y-Axis [cm]')
        drawing = True
        pos_est = []
        while drawing:
            try:
                rss = self.get_rss()
                pos_est.append(self.lambertloc(rss))
                if len(pos_est[-1]) == 1:
                    plt.plot(pos_est[-1], 0, 'bo')
                elif len(pos_est[-1]) == 2:
                    x_est = (pos_est[-1][0]**2-pos_est[-1][1]**2+dist_tx**2)/(2*dist_tx)
                    y_est = np.sqrt(pos_est[-1][0]**2 - x_est**2)
                    plt.plot(x_est, y_est, 'bo')
                plt.show()
                plt.pause(0.001)
            except KeyboardInterrupt:
                print ('Localization interrupted by user')
                drawing = False
        return pos_est

    def rfear_type(self):
        """Return a string representing the type of RfEar this is."""
        print ('LocEar,')
        print ('Number of TX: ' + str(self.__numoftx))
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
    :param results -- list containing data
    :param text -- description of results
    :param filename -- name of file (default 'Experiments')
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




