import matplotlib.pyplot as plt
from scipy.special import lambertw
import numpy as np


class EKF_Plot(object):
    def __init__(self, tx_pos, bplot_circles=True):
        """ setup figure """
        self.__tx_pos = tx_pos
        self.__tx_num = len(tx_pos)
        plt.ion()
        self.__fig1 = plt.figure(1)
        self.__ax = self.__fig1.add_subplot(111)
        self.__x1_list = []
        self.__x2_list = []

        self.__bplot_circles = bplot_circles
        self.init_plot()

    def init_plot(self):
        x_min = -500.0
        x_max = 3100.0
        y_min = -500.0
        y_max = 2000.0
        plt.axis([x_min, x_max, y_min, y_max])

        plt.grid()
        plt.xlabel('x-Axis [mm]')
        plt.ylabel('y-Axis [mm]')

        self.plot_beacons()

        if self.__bplot_circles is True:
            # init measurement circles and add them to the plot
            self.__circle_meas = []
            self.__circle_meas_est = []
            for i in range(self.__tx_num):
                txpos_single = self.__tx_pos[i]
                self.__circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='r', fill=False))
                self.__ax.add_artist(self.__circle_meas[i])
                self.__circle_meas_est.append(
                    plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='g', fill=False))
                self.__ax.add_artist(self.__circle_meas_est[i])

    def plot_beacons(self):
        # plot beacons
        for i in range(self.__tx_num):
            txpos_single = self.__tx_pos[i]
            #self.__ax.plot(txpos_single[0], txpos_single[1], 'ro')
            plt.plot(txpos_single[0], txpos_single[1], 'ro')

    def clear_plot(self):
        self.__fig1.clf()
        self.init_plot()

    def switch_b_plot_circles(self):
        if self.__bplot_circles:
            self.__bplot_circles = False
        else:
            self.__bplot_circles = True

    def add_data_to_plot(self, data_point, marker='bo'):
        self.__ax.plot(data_point[0], data_point[1], marker)
        return True

    def update_plot(self):
        # update figure 1
        self.__fig1.canvas.draw()
        plt.pause(0.001)  # pause to allow for keyboard inputs
        return True

    def plot_meas_circles(self, z_meas, y_est, tx_alpha, tx_gamma):
        numtx = len(tx_alpha)

        if self.__bplot_circles:
            for itx in range(numtx):
                #z_meas_itx = z_meas[itx]
                z_dist_itx = self.lambertloc(z_meas[itx], itx, tx_alpha, tx_gamma)
                # update measurement circles around tx-nodes
                self.__circle_meas[itx].set_radius(z_dist_itx)
                self.__circle_meas_est[itx].set_radius(y_est[itx])
        else:
            for itx in range(numtx):
                self.__circle_meas[itx].set_radius(1)
                self.__circle_meas_est[itx].set_radius(1)

    def lambertloc(self, rss, numtx, alpha, gamma):
            """Inverse function of the RSM. Returns estimated range in [mm].

            Keyword arguments:
            :param rss -- received power values [dB]
            :param numtx  -- number of the tx which rss is processed. Required to use the corresponding alpha and gamma-values.
            """
            z_dist = 20 / (np.log(10) * alpha[numtx]) * lambertw(
                np.log(10) * alpha[numtx] / 20 * np.exp(-np.log(10) / 20 * (rss + gamma[numtx])))
            return z_dist.real  # [mm]

    def plot_way_points(self, wp_list, rad_list):
        #for i in range(self.__tx_num):
            #txpos_single = self.__tx_pos[i]
            #self.__ax.plot(txpos_single[0], txpos_single[1], 'ro')
        print('wplist: ' + str(wp_list))
        x1_wp = wp_list[:, 0]
        x2_wp = wp_list[:, 1]
        plt.plot(x1_wp, x2_wp, 'go')

        num_wp = len(rad_list)
        circle_wp = []

        for i in range(num_wp):
            # txpos_single = self.__tx_pos[i]
            circle_wp.append(plt.Circle((x1_wp[i], x2_wp[i]), rad_list[i], color='g', fill=False))
            self.__ax.add_artist(circle_wp[i])
            # self.__circle_wp[i].set_radius(rad_list[i])

    def add_data_to_plot_list(self, x1, x2):
        print(str(x1[0]))
        self.__x1_list.append(x1[0])
        self.__x2_list.append(x2[0])
        #print(str(self.__x1_list))
        #print(np.shape(self.__x1_list))

    def update_live(self, numofplottedsamples=50):
        # plot data for all tx
        self.__fig1.clf()

        firstdata = 0  # set max number of plotted points per tx
        cnt = len(self.__x1_list)
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples
        #print(self.__x1_list[firstdata:-1])

        #for i in range(numoftx):
        #self.__ax.plot(self.__x_list[firstdata:-1], 'bo-',)

        plt.clf()
        firstdata = 1  # set max number of plotted points per tx
        if cnt > numofplottedsamples:
            firstdata = cnt - numofplottedsamples

        plt.plot(self.__x1_list[firstdata:-1], self.__x2_list[firstdata:-1], 'b.-', label="x_k= " + str([int(self.__x1_list[-1]), int(self.__x2_list[-1])]))

        plt.grid()
        plt.legend(loc='upper right')
        self.plot_beacons()
        plt.pause(0.001)







    """
    # init measurement circles and add them to the plot
    circle_meas = []
    circle_meas_est = []
    for i in range(EKF.get_tx_num()):
        txpos_single = EKF.get_tx_pos()[i]
        circle_meas.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='r', fill=False))
        ax.add_artist(circle_meas[i])
        circle_meas_est.append(plt.Circle((txpos_single[0], txpos_single[1]), 0.01, color='g', fill=False))
        ax.add_artist(circle_meas_est[i])

  
    """
