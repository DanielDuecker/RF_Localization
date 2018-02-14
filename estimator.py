import numpy as np
import rf

""" map/track the position of the mobile node using an EKF

    Keyword arguments:
    :param x0 -- initial estimate of the mobile node position
    :param h_func_select:
    :param bplot -- Activate/Deactivate liveplotting the data (True/False)
    :param blog -- activate data logging to file (default: False)
    :param bprintdata - activate data print to console(default: False)
    """


class ExtendedKalmanFilter(object):
    def __init__(self, set_model_type='log', x0=[1000, 1000]):
        self.__model_type = set_model_type
        self.__tx_freq = []
        self.__tx_pos = []
        self.__tx_alpha = []
        self.__tx_gamma = []

        self.__tx_freq = [4.3400e+08,   4.341e+08,   4.3430e+08,   4.3445e+08,   4.3465e+08,   4.3390e+08]

        self.__tx_pos = [[520.0, 430.0], [1540.0, 430.0], [2570.0, 430.0], [2570.0, 1230.0], [1540.0, 1230.0], [530.0, 1230.0]]

        if self.__model_type == 'log':
            """ parameter for log model """
            # self.__tx_alpha = [0.01149025464796399, 0.016245419273983631, 0.011352095690562954, 0.012125937076390217, 0.0092717529591962722, 0.01295918160582895]
            # self.__tx_gamma = [-8.5240925102693872, -11.670560994925006, -8.7169295956676116, -8.684528288347666, -5.1895194577206665, -9.8124742816198918]
            #self.__tx_alpha = [0.011642589483319907, 0.011685691737764401, 0.014099893627736611, 0.015162833988041791, 0.0083400148281428735, 0.012403542797734512]
            #self.__tx_gamma = [-7.6249148771258346, -7.570030144546223, -9.0942464068699564, -7.790184590771303, -2.1396926871992643, -6.964599659891995]

            self.__tx_alpha = [0.011100059337162281, 0.014013732682386724, 0.011873535003719441, 0.013228415946149144,
                         0.010212580857184312, 0.010286057191882235]
            self.__tx_gamma = [-0.49471304043015696, -1.2482393190627841, -0.17291318936462172, -0.61587988305564456,
                         0.99831151034040444, 0.85711994311461936]


        elif self.__model_type == 'lin':
            """ parameter for linear model """
            #self.__tx_alpha = [-0.020559673796613238, -0.027175147727451984, -0.023111068055053488, -0.024454023111282586,
            #             -0.024854213762496295, -0.021694731996127509]
            #self.__tx_gamma = [-42.189573853301056, -37.651222316888159, -40.60648965954146, -41.523883513559113,
            #             -42.342411649995938, -42.349468350676268]

        """ 
        Start RFEar as Measurement System
        """
        self.__oMeasSys = rf.RfEar(434.2e6)
        self.__oMeasSys.set_txparams(self.__tx_freq, self.__tx_pos)
        self.__oMeasSys.set_calparams(self.__tx_alpha, self.__tx_gamma)

        self.__tx_num = len(self.__tx_freq)

        """ initialize tracking setup """
        print(str(self.__tx_alpha))
        print(str(self.__tx_gamma))
        print(str(self.__tx_pos))
        self.__tx_param = []
        for itx in range(self.__tx_num):
            self.__tx_param.append([self.__tx_pos[itx], self.__tx_alpha[itx], self.__tx_gamma[itx]])

        """ initialize EKF """
        self.__x_est_0 = np.array([[x0[0]], [x0[1]]]).reshape((2, 1))
        self.__x_est = self.__x_est_0
        # standard deviations
        self.__sig_x1 = 500
        self.__sig_x2 = 500
        self.__p_mat_0 = np.array(np.diag([self.__sig_x1 ** 2, self.__sig_x2 ** 2]))
        self.__p_mat = self.__p_mat_0

        # process noise
        self.__sig_w1 = 100
        self.__sig_w2 = 100
        self.__q_mat = np.array(np.diag([self.__sig_w1 ** 2, self.__sig_w2 ** 2]))

        # measurement noise
        # --> see measurement_covariance_model
        # self.__sig_r = 10
        # self.__r_mat = self.__sig_r ** 2

        # initial values and system dynamic (=eye)
        self.__i_mat = np.eye(2)

        self.__z_meas = np.zeros(self.__tx_num)
        self.__y_est = np.zeros(self.__tx_num)
        self.__r_dist = np.zeros(self.__tx_num)

    def set_x_0(self, x0):
        self.__x_est = x0
        return True

    def set_p_mat_0(self, p0):
        self.__p_mat = p0
        return True

    def reset_ekf(self):
        self.__x_est = self.__x_est_0
        self.__p_mat = self.__p_mat_0

    def get_x_est(self):
        return self.__x_est

    def get_p_mat(self):
        return self.__p_mat

    def get_z_meas(self):
        return self.__z_meas

    def get_y_est(self):
        return self.__y_est

    def get_tx_num(self):
        return self.__tx_num

    def get_tx_pos(self):
        return self.__tx_pos

    def get_tx_alpha(self):
        return self.__tx_alpha

    def get_tx_gamma(self):
        return self.__tx_gamma

    # measurement function
    def h_rss(self, x, tx_param, model_type):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        gamma = tx_param[2]

        # r = sqrt((x - x_tx) ^ 2 + (y - y_tx) ^ 2)S
        r_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)
        if model_type == 'log':
            y_rss = -20 * np.log10(r_dist) - alpha * r_dist - gamma
        elif model_type == 'lin':
            y_rss = alpha * r_dist + gamma  # rss in db

        return y_rss, r_dist

        # Save on Raspberry

    # jacobian of the measurement function
    def h_rss_jacobian(self, x, tx_param, model_type):
        tx_pos = tx_param[0]  # position of the transceiver
        alpha = tx_param[1]
        # gamma = tx_param[2]  # not used here

        R_dist = np.sqrt((x[0] - tx_pos[0]) ** 2 + (x[1] - tx_pos[1]) ** 2)

        if model_type == 'log':
            # dh / dx1
            h_rss_jac_x1 = -20 * (x[0] - tx_pos[0]) / (np.log(10) * R_dist ** 2) - alpha * (x[0] - tx_pos[0]) / R_dist
            # dh / dx2
            h_rss_jac_x2 = -20 * (x[1] - tx_pos[1]) / (np.log(10) * R_dist ** 2) - alpha * (x[1] - tx_pos[1]) / R_dist
        elif model_type == 'lin':
            # dh /dx1
            h_rss_jac_x1 = alpha * (x[0] - tx_pos[0]) / R_dist
            # dh /dx2
            h_rss_jac_x2 = alpha * (x[1] - tx_pos[1]) / R_dist

        h_rss_jac = np.array([[h_rss_jac_x1], [h_rss_jac_x2]])

        return h_rss_jac.reshape((2, 1))

    def save_to_txt(self):

        Position = np.matrix('0 0; 0 0')
        Position[0, 0] = self.__x_est[0];
        Position[0, 1] = self.__x_est[1];
        f_EKF = open("/home/pi/src/RF_Localization/EKF.txt", "w")
        f_EKF.write(
            str(Position[0, 0]) + " " + str(Position[0, 1]) + " " + str(self.__p_mat[0, 0]) + " " + str(
                self.__p_mat[0, 1]) + " " + str(self.__p_mat[1, 0]) + " " + str(self.__p_mat[1, 1]))
        f_EKF.close
        return True

    def measurement_covariance_model(self, rss_noise_model, r_dist, itx):
        """
        estimate measurement noise based on the received signal strength
        :param rss: measured signal strength
        :return: r_mat -- measurement covariance matrix 
        """

        ekf_param = [6.5411, 7.5723, 9.5922, 11.8720, 21.6396, 53.6692, 52.0241]
        #if r_dist <= 120 or r_dist >= 1900:
        #        r_sig = 100
        #
        #         print('r_dist = ' + str(r_dist))
        #rss_max_lim = [-42, -42, -42, -42, -42, -42]*0
        if -35 < rss_noise_model or r_dist >= 1900:
            r_sig = 100
            #print('Meas_Cov_Model: itx = ' + str(itx) + ' r_sig set to ' + str(r_sig))

        else:
            if rss_noise_model >= -55:
               r_sig = ekf_param[0]
            elif rss_noise_model < -55:
                r_sig = ekf_param[1]
            elif rss_noise_model < -65:
                 r_sig = ekf_param[2]
            elif rss_noise_model < -75:
                r_sig = ekf_param[3]
            elif rss_noise_model < -80:
                r_sig = ekf_param[4]


        r_mat = r_sig ** 2
        return r_mat
       #
        """
        #old model
        # simple first try
        if rss_noise_model >= -85:
            r_sig = 10
        elif rss_noise_model < -85:
            r_sig = 100

        r_mat = r_sig ** 2
        return r_mat
        
        """

    def ekf_prediction(self):
        """ prediction """
        self.__x_est = self.__x_est  # + np.random.randn(2, 1) * 1  # = I * x_est
        self.__p_mat = self.__i_mat.dot(self.__p_mat.dot(self.__i_mat)) + self.__q_mat
        return True



    def ekf_update(self,rss_low_lim=-120):
        """ innovation """

        freq_peaks, rss = self.__oMeasSys.get_rss_peaks()
        # get new measurement
        self.__z_meas = rss

        """ if no valid measurement signal is received, reset ekf i.e. boat outside water"""
        if np.max(rss) < rss_low_lim:
            self.reset_ekf()
            print('reset_ekf'+str(np.max(rss)))
            print('rss'+str(rss))
            return True

        # iterate through all tx-rss-values
        for itx in range(self.__tx_num):
            # estimate measurement from x_est
            self.__y_est[itx], self.__r_dist[itx] = self.h_rss(self.__x_est, self.__tx_param[itx], model_type=self.__model_type)
            y_tild = self.__z_meas[itx] - self.__y_est[itx]

            # estimate measurement noise based on
            r_mat = self.measurement_covariance_model(self.__z_meas[itx], self.__r_dist[itx], itx)

            # calc K-gain
            h_jac_mat = self.h_rss_jacobian(self.__x_est, self.__tx_param[itx], model_type=self.__model_type)
            s_mat = np.dot(h_jac_mat.transpose(), np.dot(self.__p_mat, h_jac_mat)) + r_mat  # = H^t * P * H + R
            k_mat = np.dot(self.__p_mat, h_jac_mat / s_mat)  # 1/s_scal since s_mat is dim = 1x1

            self.__x_est = self.__x_est + k_mat * y_tild  # = x_est + k * y_tild
            self.__p_mat = (self.__i_mat - np.dot(k_mat, h_jac_mat.transpose())) * self.__p_mat  # = (I-KH)*P
        return True

    def check_valid_position_estimate(self,x_field_begin=[-500 ,-500], x_field_end=[3700, 2600]):
        if x_field_begin[0] > self.__x_est[0] or x_field_end[0] < self.__x_est[0]:
            self.reset_ekf()
            print('EKF: Position estimate out of range --> reset EKF')
        elif x_field_begin[1] > self.__x_est[1] or x_field_end[1] < self.__x_est[1]:
            self.reset_ekf()
            print('EKF: Position estimate out of range --> reset EKF')
        return True


"""

EKF = ExtendedKalmanFilter()
EKF_plotter = EKF_Plot(EKF.get_tx_pos(), EKF.get_tx_num())


# set EKF init position

x_log = np.array([[500], [500]])
EKF.init_x_0(x_log)

### Start EKF-loop ###
tracking = True
while tracking:
    try:
        # x_est[:, 0] = x_log[:, -1]
        EKF.ekf_prediction()
        EKF.ekf_update()

        x_log = np.append(x_log, EKF.get_x_est(), axis=1)

        # add new x_est to plot
        EKF_plotter.add_data_to_plot([EKF.get_x_est()[0, -1], EKF.get_x_est()[1, -1]])
        EKF_plotter.update_plot()

    except KeyboardInterrupt:
        print ('Localization interrupted by user')
        tracking = False

"""