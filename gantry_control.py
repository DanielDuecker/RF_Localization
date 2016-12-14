import numpy as np
import matplotlib.pyplot as plt
import time as t
import rf


class GantryControl(object):
    def __init__(self, gantry_dimensions=[0, 2500, 0, 4500]):  # [x0 ,x1, y0, y1]
        self.__dimensions = gantry_dimensions
        self.__gantry_pos = [0, 0]  # initial position after start
        self.__target_wp = []
        self.__oCal = []
        self.__oLoc = []

    def get_gantry_dimensions(self):
        return self.__dimensions

    def get_gantry_pos(self):
        return self.__gantry_pos

    def set_gantry_pos(self, new_pos):
        self.__gantry_pos = new_pos

    def set_target_wp(self, target_wp):
        if len(target_wp) == len(self.__gantry_pos):
            if self.check_wp_in_workspace(target_wp):
                self.__target_wp = target_wp
                b_new_wp = True
            else:
                print('ERROR:target way-point: x=' + str(target_wp(0) + ' y=' + str(target_wp(1)) + ' not in workspace'))
                b_new_wp = False
        else:
            print('ERROR: Dimension mismatch!')
            print('len(target_wp) ='+str(len(target_wp))+' ~= len(self.__gantry_pos)  ='+str(len(self.__gantry_pos)))
            b_new_wp = False
        return b_new_wp

    def check_wp_in_workspace(self, wp):
        gantry_dim = self.get_gantry_dimensions()

        if wp[0] >= gantry_dim[0] and wp[0] <= gantry_dim[1] and wp[1] >= gantry_dim[2] and wp[1] <= gantry_dim[3]:
            bvalid_wp = True
        else:
            print ('ERROR: Target way-point cannot be approached!')
            print ('Target way-point ' + str(wp) + ' does not lie within the gantry workspace ' +
                   'x= [' + str(gantry_dim[0]) + ' ... ' + str(gantry_dim[2]) + '], ' +
                   'y= [' + str(gantry_dim[1]) + ' ... ' + str(gantry_dim[3]) + '] ')
            bvalid_wp = False
        return bvalid_wp

    def transmit_wp_to_gantry(self, targetwp):
        if self.set_target_wp(targetwp):
            btransmission = True
        else:
            print ('ERROR: wp-transmission to gantry failed!')
            btransmission = False
        return btransmission

    def move_gantry_to_target(self):
        target_wp = self.__target_wp

        print ('move gantry to way-point x [mm] = ' + str(target_wp[0]) + ' y [mm] = ' + str(target_wp[1]))


        # some control stuff can be inserted here
        t.sleep(1.0)
        self.set_gantry_pos(target_wp)
        if self.get_gantry_pos() == target_wp:
            print ('arrived at new waypoint')
            inc_pos = target_wp[0] * 1e6 / 310 + 1e6
            raw_input(' confirm arrival at INC pos x [inc] = ' + str(inc_pos))
            bArrived = True
        else:
            print ('Gantry haven t arrived at target way-point')
            actual_wp = self.get_gantry_pos()
            print ('Actual position: x= ' + str(actual_wp[0]) + ' y= ' + str(actual_wp[1]) +
                   ' target wp x= ' + str(target_wp[0]) + ' y= ' + str(target_wp[1]))
            bArrived = False

        return bArrived

    def confirm_arrived_at_wp(self):
        barrivalconfirmed = False
        if self.__target_wp == self.__gantry_pos:
            barrivalconfirmed = True

        return barrivalconfirmed

    def process_measurement_sequence(self, wplist_filename, measdata_filename):
        """
        #
        :param wplist_filename:
        :param measdata_filename:
        :return:
        """
        # read data from waypoint file
        with open(wplist_filename, 'r') as wpfile:
            wp_data_list = [map(float, line.split(',')) for line in wpfile]
            wp_data_mat = np.asarray(wp_data_list)
            wpfile.close()

        with open(measdata_filename, 'w') as measfile:
            # write header to measurement file

            measfile.write(t.ctime() + '\n')
            measfile.write('some description' + '\n')
            measfile.write('\n')

            # setup plot
            plt.ion()
            plt.plot(wp_data_mat[:, 1], wp_data_mat[:, 2], 'b.-')
            plt.xlabel('Distance in mm')
            plt.ylabel('Distance in mm')
            plt.xlim(-10, 1800)
            plt.ylim(-10, 2000)
            plt.grid()
            plt.show()

            totnumofwp = np.shape(wp_data_mat)

            totnumofwp = totnumofwp[0]
            print ('Number of waypoints = ' + str(totnumofwp) + '\n')

            # loop over all way-points
            for row in wp_data_mat:

                numwp = int(row[0])
                new_target_wpx = row[1]
                new_target_wpy = row[2]
                new_target_wp = [new_target_wpx, new_target_wpy]  # find a solution for this uggly workaround...
                meastime = row[3]

                if self.transmit_wp_to_gantry(new_target_wp):
                    if self.move_gantry_to_target():
                        if self.confirm_arrived_at_wp():
                            print('START Measurement for ' + str(meastime) + 's')
                            print('Measuring at Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points')
                            plt.plot(new_target_wp[0], new_target_wp[1], 'go')
                            plt.title('Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points')
                            dataseq = self.__oCal.take_measurement(meastime, outputmode='fft')

                            [nummeas, numtx] = np.shape(dataseq)

                            # way point data - structure 'wp_x, wp_y, num_wp, num_tx, num_meas'
                            str_base_data = str(new_target_wp[0]) + ', ' + str(new_target_wp[1]) + ', ' +\
                                            str(numwp) + ', ' + str(numtx) + ', ' + str(nummeas) + ', '
                            # freq data
                            freqs = self.__oCal.get_freq()
                            str_freqs = ', '.join(map(str, freqs)) + ', '

                            # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
                            #print('data ' + str(dataseq))
                            str_rss = ''
                            for i in range(numtx):
                                str_rss = str_rss + ', '.join(map(str, dataseq[:, i])) + ', '

                            measfile.write(str_base_data + str_freqs + str_rss + '\n')
                            # print(str_base_data + str_freqs + str_rss)

                    else:
                        print ('Error: Failed to move gantry to new way-point!')
                        print ('Way-point #' + str(numwp) + ' @ position x= ' +
                               str(new_target_wp[0]) + ', y = ' + str(new_target_wp[1]))

                else:
                    print ('Error: Failed to transmit new way-point to gantry!')
                    print ('point# ' + str(numwp) + ' @ position x= ' +
                           str(new_target_wp[0]) + ', y = ' + str(new_target_wp[1]))
                plt.pause(0.001)
                print
            measfile.close()
        return True

    def start_CalEar(self, freqtx=433.9e6, freqspan=2e4):
        self.__oCal = rf.CalEar(freqtx, freqspan)
        return True

    def start_LocEar(self, alpha, xi, txpos, freqtx, freqspan=2e4):
        self.__oLoc = rf.LocEar(alpha, xi, txpos, freqtx, freqspan)
        return True

"""
independent methods related to the gantry
"""


def wp_generator(wp_filename='wplist.txt', x0=[0, 0], xn=[1200, 1200], steps=[7, 7], timemeas=10.0):
    """
    :param wp_filename:
    :param x0: [x0,y0] - start position of the grid
    :param xn: [xn,yn] - end position of the grid
    :param steps: [numX, numY] - step size
    :param timemeas: - time [s] to wait at each position for measurements
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

    with open(wp_filename, 'w') as wpfile:
        # wpfile.write(t.ctime() + '\n')
        # wpfile.write('some describtion' + '\n')
        for i in range(wp_mat.shape[0]):
            wpfile.write(str(i) + ', ' + str(wp_mat[i, 0]) + ', ' + str(wp_mat[i, 1]) + ', ' + str(wp_mat[i, 2]) + '\n')
        wpfile.close()

    plt.figure()
    plt.plot(wp_mat[:, 0], wp_mat[:, 1], '.-')
    plt.show()

    return wp_filename  # file output [line#, x, y, time]
