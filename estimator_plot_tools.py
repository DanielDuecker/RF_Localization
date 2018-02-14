import matplotlib.pyplot as plt
#from matplotlib.patches import Arrow
from scipy.special import lambertw
import numpy as np
import time as t


class EKF_Plot(object):
    def __init__(self, tx_pos, model_type='log', bplot_circles=True, b_p_cov_plot=False):
        """ setup figure """
        self.__tx_pos = tx_pos
        self.__tx_num = len(tx_pos)
        self.__rsm_model_type = model_type
        plt.ion()
        (self.__fig1, self.__ax1) = plt.subplots()  # get figure/axis handles
        if b_p_cov_plot:
            (self.__fig2, self.__ax2) = plt.subplots()  # get figure/axis handles

        self.__act_time = 0.0  # time to calc plot update frequency

        self.__x1_list = []
        self.__x2_list = []
        self.__x1_gantry_list = []
        self.__x2_gantry_list = []
        self.__p11_list = []
        self.__p22_list = []

        self.__bplot_circles = bplot_circles
        self.init_plot(b_p_cov_plot)

    def init_plot(self, b_cov_plot):
        if b_cov_plot:
            # self.__ax2.set_xlabel
            # self.__ax2.set_ylabel
            self.__ax2.grid()
        x_min = -500.0
        x_max = 3800.0
        y_min = -500.0
        y_max = 2000.0
        self.__ax1.axis([x_min, x_max, y_min, y_max])
        self.__ax1.axis('equal')

        self.__ax1.grid()
        self.__ax1.set_xlabel('x-Axis [mm]')
        self.__ax1.set_ylabel('y-Axis [mm]')

        tank_frame = np.array([[-250, -150], [3750, -150], [3750, 1850], [-250, 1850], [-250, -150]])
        self.__ax1.plot(tank_frame[:, 0], tank_frame[:, 1], 'k-')

        self.plot_beacons()

        if self.__bplot_circles is True:
            # init measurement circles and add them to the plot
            self.__circle_meas = []
            self.__circle_meas_est = []
            for i in range(self.__tx_num):
                txpos_single = self.__tx_pos[i]
                self.__circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.1, color='r', fill=False))
                self.__ax1.add_artist(self.__circle_meas[i])
                self.__circle_meas_est.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.1, color='g', fill=False))
                self.__ax1.add_artist(self.__circle_meas_est[i])

    def plot_beacons(self):
        # plot beacons
        for i in range(self.__tx_num):
            txpos_single = self.__tx_pos[i]
            self.__ax1.plot(txpos_single[0], txpos_single[1], 'ko')



    def plot_way_points(self, wp_list=np.array([0,0]), wp_rad=[0], b_plot_circles=False):
        x1_wp = wp_list[:, 0]
        x2_wp = wp_list[:, 1]
        self.__ax1.plot(x1_wp, x2_wp, 'g.-', label="Way - Point")
        num_wp = len(wp_rad)
        circle_wp = []
        if b_plot_circles:
            for i in range(num_wp):

                circle_wp.append(plt.Circle((x1_wp[i], x2_wp[i]), wp_rad[i], color='g', fill=False))
                self.__ax1.add_artist(circle_wp[i])

    def update_meas_circles(self, z_meas, alpha, gamma, b_plot_yest=False, y_est=[], rsm_model='log'):
        """

        :param z_meas:
        :param b_plot_yest:
        :param y_est:
        :param alpha:
        :param gamma:
        :return:
        """
        for itx in range(self.__tx_num):
            z_dist = self.inverse_rsm(z_meas[itx], alpha[itx], gamma[itx], self.__rsm_model_type)
            self.__circle_meas[itx].set_radius(z_dist)
            if b_plot_yest:
                z_est = self.inverse_rsm(y_est[itx], alpha[itx], gamma[itx], self.__rsm_model_type)
                self.__circle_meas_est[itx].set_radius(z_est)

                #print('y_tild=' + str(z_meas-y_est))

    def inverse_rsm(self, rss, alpha, gamma, rsm_model_type):
            """Inverse function of the RSM. Returns estimated range in [mm].

            Keyword arguments:
            :param rss -- received power values [dB]
            :param alpha
            :param gamma
            :param rsm_model_type
            """
            #if rsm_model_type == 'log':
            z_dist = 20 / (np.log(10) * alpha) * lambertw(
                    np.log(10) * alpha / 20 * np.exp(-np.log(10) / 20 * (rss + gamma)))
            #elif rsm_model_type == 'lin':
            #    z_dist = (rss - gamma) / alpha

            return z_dist.real  # [mm]

    def add_x_est_to_plot(self, x_est):
        self.__x1_list.append(x_est[0])
        self.__x2_list.append(x_est[1])

    def plot_gantry_pos(self, x_gantry):
        self.__x1_gantry_list.append(x_gantry[0])
        self.__x2_gantry_list.append(x_gantry[1])

    def add_p_cov_to_plot(self, p_mat):
        self.__p11_list.append(np.sqrt(p_mat[0, 0]))
        self.__p22_list.append(np.sqrt(p_mat[1, 1]))

    def plot_p_cov(self,numofplottedsamples=300):
        firstdata = 0  # set max number of plotted points
        cnt = len(self.__x1_list)
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples
        if len(self.__p11_list) > 1:
            del self.__ax2.lines[-1]
            del self.__ax2.lines[-1]

        self.__ax2.plot(self.__p11_list[firstdata:-1], 'b.-', label='P11-std')
        self.__ax2.plot(self.__p22_list[firstdata:-1], 'r.-', label='P22-std')
        self.__ax2.legend(loc='upper right')

        plt.pause(0.001)


    def plot_ekf_pos_live(self, b_plot_gantry=False, numofplottedsamples=50):
        """
        This function must be the last plot function due to the ugly 'delete' workaround
        :param numofplottedsamples:
        :return:
        """
        new_time = float(t.time())
        self.__ax1.set_title('Vehicle postition [update with ' + str(int(1 / (new_time - self.__act_time))) + ' Hz]')
        self.__act_time = new_time


        firstdata = 0  # set max number of plotted points
        cnt = len(self.__x1_list)
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples
        if len(self.__x1_list) > 1:
            del self.__ax1.lines[-1]
            del self.__ax1.lines[-1]
            if b_plot_gantry:
                del self.__ax1.lines[-1]

        self.__ax1.plot(self.__x1_list[firstdata:-1], self.__x2_list[firstdata:-1], 'b.-')
        self.__ax1.plot(self.__x1_list[-1], self.__x2_list[-1], 'ro',
                        label="x_k= " + str([int(self.__x1_list[-1]), int(self.__x2_list[-1])]))
        if b_plot_gantry:
            self.__ax1.plot(self.__x1_gantry_list, self.__x2_gantry_list, 'go-',
                            label="x_k= " + str([int(self.__x1_gantry_list[-1]), int(self.__x2_gantry_list[-1])]))
        self.__ax1.legend(loc='upper right')

        plt.pause(0.001)

    def plot_yaw_angle_live(self, yaw_angle_rad=0):
        x1_pos = self.__x1_list[-1]
        x2_pos = self.__x2_list[-1]
        if len(self.__x1_list) > 1:
            self.__yaw_arrow.remove()

        self.__yaw_arrow = self.__ax1.arrow(x1_pos[0], x2_pos[0], 400 * np.cos(yaw_angle_rad), 400 * np.sin(yaw_angle_rad),
                                            head_width=40, head_length=150, fc='k', ec='k')

    #def plot_next_target_wp(self, next_wp):
        #if len(self.__x1_list) > 1:
        #    self.__yaw_arrow.remove()
        #self.__ax1.plot(next_wp[0], next_wp[1], 'go',
        #                label="x_wp= " + str([int(next_wp[0]), int(next_wp[1])]))
