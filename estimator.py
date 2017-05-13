# import modules
from abc import ABCMeta, abstractmethod
import time as t
import numpy as np
import matplotlib.pyplot as plt
import rf


def map_path_ekf(x0, h_func_select='h_rss', bplot=True, blog=False, bprintdata=False):
    """ map/track the position of the mobile node using an EKF

    Keyword arguments:
    :param x0 -- initial estimate of the mobile node position
    :param h_func_select:
    :param bplot -- Activate/Deactivate liveplotting the data (True/False)
    :param blog -- activate data logging to file (default: False)
    :param bprintdata - activate data print to console(default: False)
    """
    tx_freq = []
    tx_pos = []
    tx_alpha = []
    tx_gamma = []

    tx_freq = [434.00e6, 434.15e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

    tx_pos = [[790, 440],
               [1650, 450],
               [2530, 460],
               [2530, 1240],
               [1650, 1235],
               [790, 1230]]
    tx_alpha = [0.013854339628529109,
                0.0071309466013866158,
                0.018077579531274993,
                0.016243668091798915,
                0.016243668091798915,
                0.016243668091798915]
    tx_gamma = [-1.5898021024559508,
                2.0223747861988461,
                -5.6650866655302714,
                -5.1158161676293972,
                -5.1158161676293972,
                -5.1158161676293972]

    oMeasSys = rf.RfEar(434.2e6)
    oMeasSys.set_txparams(tx_freq, tx_pos)
    oMeasSys.set_calparams(tx_alpha, tx_gamma)

    tx_num = len(tx_freq)

    # measurement function
    def h_rss(x, tx_param):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        gamma = tx_param[2]

        # r = sqrt((x - x_tx) ^ 2 + (y - y_tx) ^ 2)
        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)
        y_rss = -20 * np.log10(r_dist) - alpha * r_dist - gamma

        return y_rss

    # jacobian of the measurement function
    def h_rss_jacobian(x_est, tx_param):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        # gamma = tx_param[2]  # not used here

        R_dist = np.sqrt((x_est[0] - tx_pos[0]) ** 2 + (x_est[1] - tx_pos[1]) ** 2)

        # dh / dx1
        h_rss_jac_x = -20 * (x_est[0] - tx_pos[0]) / (np.log(10) * R_dist ** 2) - alpha * (
        x_est[0] - tx_pos[0]) / R_dist
        # dh / dx2
        h_rss_jac_y = -20 * (x_est[1] - tx_pos[1]) / (np.log(10) * R_dist ** 2) - alpha * (
        x_est[1] - tx_pos[1]) / R_dist

        h_rss_jac = np.array([[h_rss_jac_x], [h_rss_jac_y]])

        return h_rss_jac

    def measurement_covariance_model(rss_noise_model):
        """
        estimate measurement noise based on the received signal strength
        :param rss: measured signal strength
        :return: r_mat -- measurement covariance matrix 
        """

        # simple first try
        if rss_noise_model >= -85:
            r_sig = 10
        elif rss_noise_model < -85:
            r_sig = 100

        r_mat = r_sig ** 2
        return r_mat

    if h_func_select == 'h_rss':
        h = h_rss
        h_jacobian = h_rss_jacobian
    else:
        print ('You need to select to a measurement function "h" like "h_rss" or "h_dist"!')
        print ('exit...')
        return True

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

        for i in range(tx_num):
            txpos_single = tx_pos[i]
            ax.plot(txpos_single[0], txpos_single[1], 'ro')

        # init measurement circles and add them to the plot
        circle_meas = []
        circle_meas_est = []
        for i in range(tx_num):
            txpos_single = tx_pos[i]
            circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='r', fill=False))
            ax.add_artist(circle_meas[i])
            circle_meas_est.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='g', fill=False))
            ax.add_artist(circle_meas_est[i])

    """ initialize tracking setup """
    print(str(tx_alpha))
    print(str(tx_gamma))
    print(str(tx_pos))
    tx_param = []
    for itx in range(tx_num):
        tx_param.append([tx_pos[itx], tx_alpha[itx], tx_gamma[itx]])

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
    # --> see measurement_covariance_model
    #sig_r = 10
    #r_mat = sig_r ** 2

    # initial values and system dynamic (=eye)
    x_log = np.array([[x0[0]], [x0[1]]])
    x_est = x_log

    i_mat = np.eye(2)

    z_meas = np.zeros(tx_num)
    y_est = np.zeros(tx_num)

    """ Start EKF-loop """
    tracking = True
    while tracking:
        try:
            x_est[:, 0] = x_log[:, -1]

            """ prediction """
            x_est = x_est  # + np.random.randn(2, 1) * 1  # = I * x_est
            p_mat_est = i_mat.dot(p_mat.dot(i_mat)) + q_mat

            """ innovation """
            # get new measurement
            freq_peaks, rss = oMeasSys.get_rss_peaks()
            z_meas = rss

            # iterate through all tx-rss-values
            for itx in range(tx_num):
                # estimate measurement from x_est
                y_est[itx] = h(x_est, tx_param[itx])
                y_tild = z_meas[itx] - y_est[itx]

                # estimate measurement noise based on
                r_mat = measurement_covariance_model(z_meas[itx])

                # calc K-gain
                h_jac_mat = h_jacobian(x_est[:, 0], tx_param[itx])
                s_mat = np.dot(h_jac_mat.transpose(), np.dot(p_mat, h_jac_mat)) + r_mat  # = H^t * P * H + R
                k_mat = np.dot(p_mat, h_jac_mat / s_mat)  # 1/s_scal since s_mat is dim = 1x1

                x_est = x_est + k_mat * y_tild  # = x_est + k * y_tild
                p_mat = (i_mat - np.dot(k_mat, h_jac_mat.transpose())) * p_mat_est  # = (I-KH)*P
            x_log = np.append(x_log, x_est, axis=1)

            """ update figure / plot after all measurements are processed """
            if bplot:
                # add new x_est to plot
                ax.plot(x_est[0, -1], x_est[1, -1], 'bo')
                # update measurement circles around tx-nodes
                """
                for i in range(tx_num):
                    circle_meas[i].set_radius(z_meas[i])
                    circle_meas_est[i].set_radius(y_est[i])
                """
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

map_path_ekf([1000, 1000])
