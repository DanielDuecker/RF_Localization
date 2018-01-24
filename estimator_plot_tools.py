import matplotlib.pyplot as plt


class EKF_Plot(object):
    def __init__(self, tx_pos):
        """ setup figure """
        self.__tx_pos = tx_pos
        self.__tx_num = len(tx_pos)
        plt.ion()
        self.__fig1 = plt.figure(1)
        self.__ax = self.__fig1.add_subplot(111)
        self.__x_list = []

        x_min = -500.0
        x_max = 3100.0
        y_min = -500.0
        y_max = 2000.0
        plt.axis([x_min, x_max, y_min, y_max])

        plt.grid()
        plt.xlabel('x-Axis [mm]')
        plt.ylabel('y-Axis [mm]')

        self.plot_beacons()

    def plot_beacons(self):
        # plot beacons
        for i in range(self.__tx_num):
            txpos_single = self.__tx_pos[i]
            self.__ax.plot(txpos_single[0], txpos_single[1], 'ro')

    def add_data_to_plot(self, data_point, marker='bo'):
        self.__ax.plot(data_point[0], data_point[1], marker)
        return True

    def update_plot(self):
        # update figure 1
        self.__fig1.canvas.draw()
        plt.pause(0.001)  # pause to allow for keyboard inputs
        return True

    """
        def add_data_to_plot_list(self, x1, x2):
            self.__x_list.append([x1, x2])
    
        def update_live(self, numofplottedsamples=250):
            # plot data for all tx
            self.__fig1.clf()
    
            firstdata = 1  # set max number of plotted points per tx
            cnt = len(self.__x_list)
            if cnt > numofplottedsamples:
                firstdata = cnt - numofplottedsamples
            print(self.__x_list[firstdata:-1])
            #for i in range(numoftx):
            #self.__ax.plot(self.__x_list[firstdata:-1], 'bo-',)
            self.__ax.grid()
            self.plot_beacons()
            plt.pause(0.001)
    
    """


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

    # update measurement circles around tx-nodes

            for i in range(tx_num):
                circle_meas[i].set_radius(z_meas[i])
                circle_meas_est[i].set_radius(y_est[i])   
    """
