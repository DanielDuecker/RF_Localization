import numpy as np
import matplotlib.pyplot as plt
import time as t
import serial_control as sc
import hippocampus_toolbox as hc_tools


class GantryControl(object):
    def __init__(self, gantry_dimensions=[0, 3000, 0, 1580], use_gui=False):  # [x0 ,x1, y0, y1]
        self.__dimensions = gantry_dimensions
        self.__gantry_pos = [0, 0]  # initial position after start
        self.__target_wp_mm = []
        self.__oCal = []
        self.__oLoc = []
        self.__oScX = []  # spindle-drive
        self.__oScY = []  # belt-drive
        self.__maxposdeviation = 2  # [mm] max position deviation per axis

        self.__oScX = sc.MotorCommunication('/dev/ttyS4', 'belt_drive', 'belt', 3100, 2000e3)
        self.__oScY = sc.MotorCommunication('/dev/ttyS5', 'spindle_drive', 'spindle', 1600, 5150e3)

        self.__starttime = []

        if use_gui:
            print('Gantry Control - gui mode')
            self.__oScX.open_port()
            self.__oScY.open_port()
        else:
            # belt-drive
            self.__oScX.open_port()
            self.__oScX.start_manual_mode()
            self.__oScX.enter_manual_init_data()
            if self.__oScX.get_manual_init() is False:
                self.__oScX.initialize_home_pos()
                #self.__oScX.initialize_extreme_pos()
            print('Belt-Drive: Setup DONE!')

            # spindle-drive
            self.__oScY.open_port()
            self.__oScY.start_manual_mode()
            self.__oScY.enter_manual_init_data()
            if self.__oScY.get_manual_init is False:
                self.__oScY.initialize_home_pos()
                #self.__oScY.initialize_extreme_pos()
            print('Spindle-Drive: Setup DONE!')

    def get_serial_x_handle(self):
        return self.__oScX

    def get_serial_y_handle(self):
        return self.__oScY

    def get_gantry_dimensions(self):
        return self.__dimensions

    def get_gantry_pos(self):
        return self.__gantry_pos

    def set_starttime(self):
        self.__starttime = t.ctime()

    def get_starttime(self):
        return self.__starttime

    def set_gantry_pos(self, new_pos):
        self.__gantry_pos = new_pos

    def set_target_wp(self, target_wp):
        if len(target_wp) == len(self.__gantry_pos):
            if self.check_wp_in_workspace(target_wp):
                self.__target_wp_mm = target_wp
                b_new_wp = True
            else:
                print('ERROR:target way-point: x=' + str(target_wp(0) + ' y=' + str(target_wp(1)) + ' not in workspace'))
                b_new_wp = False
        else:
            print('ERROR: Dimension mismatch!')
            print('len(target_wp) ='+str(len(target_wp))+' ~= len(self.__gantry_pos)  ='+str(len(self.__gantry_pos)))
            b_new_wp = False
        return b_new_wp

    def get_target_wp_mm(self):
        return self.__target_wp_mm

    def get_gantry_pos_xy_mm(self):
        pos_x_mm = self.__oScX.get_posmm()  # belt-drive
        pos_y_mm = self.__oScY.get_posmm()  # spindle-drive
        return pos_x_mm, pos_y_mm

    def start_go_home_seq_xy(self):
        self.__oScX.start_home_seq()  # belt-drive
        self.__oScY.start_home_seq()  # spindle-drive

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

    def confirm_arrived_at_target_wp(self):
        """
        This method checks whether the gantry has arrived at its target position
        within a range of 'maxdeviation' [mm]
        :return: flag - arrived true/false
        """
        barrival_confirmed = False
        gantry_pos_mm = self.get_gantry_pos_xy_mm()
        target_pos_mm = self.get_target_wp_mm()
        distx = abs(gantry_pos_mm[0] - target_pos_mm[0])
        disty = abs(gantry_pos_mm[1] - target_pos_mm[1])

        if distx < self.__maxposdeviation and disty < self.__maxposdeviation:
            barrival_confirmed = True

        return barrival_confirmed

    def start_moving_gantry_to_target(self):
        target_wp = self.get_target_wp_mm()

        print ('Move gantry to way-point x [mm] = ' + str(target_wp[0]) + ' y [mm] = ' + str(target_wp[1]))
        self.__oScX.go_to_pos_mm(target_wp[0])
        self.__oScY.go_to_pos_mm(target_wp[1])

    def move_gantry_to_target(self):
        self.start_moving_gantry_to_target()

        bArrived_both = False
        while bArrived_both is False:
            t.sleep(0.1)

            actpos_X_mm, actpos_Y_mm = self.get_gantry_pos_xy_mm()
            actpos = [actpos_X_mm, actpos_Y_mm]
            print('Actual position: x=' + str(actpos[0]) + 'mm y=' + str(actpos[1]) + 'mm')
            self.set_gantry_pos(actpos)
            """
            dist_x = abs(self.get_gantry_pos()[0] - target_wp[0])
            dist_y = abs(self.get_gantry_pos()[1] - target_wp[1])
            if dist_x < self.__maxposdeviation and dist_y < self.__maxposdeviation:
                print ('Arrived at way-point')
                bArrived_both = True
            """
            if self.confirm_arrived_at_target_wp():
                print ('Arrived at way-point')
                bArrived_both = True

        # @todo: position feedback from motor  check whether position is within a suitable range

        return bArrived_both

    def move_gantry_to_target_manual(self):
        target_wp = self.get_target_wp_mm()

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

    def follow_wp_and_take_measurements(self):
        start_wp = [1500, 600]
        wp_list = [[2200, 600],
                   [2200, 1000],
                   [1500, 1000]]
        num_wp = len(wp_list)
        print('Number of way points: ' + str(num_wp))
        start_time = t.time()

        self.set_target_wp(start_wp)
        self.start_moving_gantry_to_target()
        while not self.confirm_arrived_at_target_wp():
            print('Moving to start position = ' + str(start_wp))
            t.sleep(0.2)

        for wp in wp_list:
            # got to wp
            self.set_target_wp(wp)
            self.start_moving_gantry_to_target()
            not_arrived_at_wp = True
            print('Moving to wp = ' + str(wp))
            while not_arrived_at_wp:
                time_elapsed = t.time() - start_time
                pos_x_mm, pos_y_mm = self.get_gantry_pos_xy_mm()
                freq_den_max, pxx_den_max = self.__oCal.get_rss_peaks_from_single_sample()
                print('Time = ' + str(time_elapsed) + ' Pos = [' + str(pos_x_mm) + ', ' + str(pos_y_mm) + ']: peaks = '
                      + str(pxx_den_max))
                if self.confirm_arrived_at_target_wp():
                    not_arrived_at_wp = False



    def follow_wp_trajectory(self, vdes_x, vdes_y, dist_threshhold):
        """

        :param vdes_x:
        :param vdes_y:
        :param dist_threshhold: [mm]
        :return:
        """

        wp_list = [[1500, 600],
                   [2200, 600],
                   [2200, 1000],
                   [1500, 1000],
                   [1500, 600]]

        num_wp = len(wp_list)
        print('Number of way points: ' + str(num_wp))
        """
        insert a sequence to move to starting point
        """

        for i_wp in range(num_wp):

            target_wp = wp_list[i_wp]
            print('Target_wp = ' + str(target_wp))
            self.set_target_wp(target_wp)
            bdrive_x_arrived = False
            bdrive_y_arrived = False

            if self.__oScX.get_dist_to(target_wp[0]) >= 0:
                v_x = vdes_x
            elif self.__oScX.get_dist_to(target_wp[0]) < 0:  # move backwards
                v_x = -vdes_x

            if self.__oScY.get_dist_to(target_wp[1]) >= 0:
                v_y = vdes_y
            elif self.__oScY.get_dist_to(target_wp[1]) < 0:  # move backwards
                v_y = -vdes_y

            self.__oScX.set_drive_speed(v_x)
            self.__oScY.set_drive_speed(v_y)

            target_pos_reached = False
            while target_pos_reached is False:
                print('X-dist = ' + str(self.__oScX.get_dist_to(target_wp[0])) + ' speed: ' + str(v_x))
                print('Y-dist = ' + str(self.__oScY.get_dist_to(target_wp[1])) + ' speed: ' + str(v_y))

                if abs(self.__oScX.get_dist_to(target_wp[0])) < dist_threshhold:
                    v_x = 0
                    self.__oScX.set_drive_speed(v_x)
                    bdrive_x_arrived = True
                if abs(self.__oScY.get_dist_to(target_wp[1])) < dist_threshhold:
                    v_y = 0
                    self.__oScY.set_drive_speed(v_y)
                    bdrive_y_arrived = True

                if bdrive_x_arrived and bdrive_y_arrived:
                    print('Arrived!')
                    target_pos_reached = True
            print('Go to next WP')

        return True

    def process_measurement_sequence(self, wplist_filename, measdata_filename):
        """
        :return:
        """


        print('Process Measurement Sequence started')
        # read data from waypoint file
        #wplist_filename = hc_tools.select_file()

        with open(wplist_filename, 'r') as wpfile:
            wp_data_list = [map(float, line.split(',')) for line in wpfile]
            wp_data_mat = np.asarray(wp_data_list)
            wpfile.close()

        #measdata_filename = hc_tools.save_as_dialog('Save measurement data as...')
        with open(measdata_filename, 'w') as measfile:
            # write header to measurement file

            measfile.write(t.ctime() + '\n')
            measfile.write('some description' + '\n')
            measfile.write('\n')

            # setup plot
            plt.ion()
            plt.plot(wp_data_mat[:, 1], wp_data_mat[:, 2], 'b.-')
            plt.xlabel('Distance in mm (belt-drive)')
            plt.ylabel('Distance in mm (spindle-drive)')
            plt.xlim(-10, 2940)
            plt.ylim(-10, 1700)
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
                        if self.confirm_arrived_at_target_wp():
                            t.sleep(1)  # wait to damp motion/oscillation of antenna etc

                            print('START Measurement for ' + str(meastime) + 's')
                            print('Measuring at Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points')
                            plt.plot(new_target_wp[0], new_target_wp[1], 'go')
                            plt.title('Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points ' +
                                      '- measurement sequence was started at ' + str(self.get_starttime()))
                            dataseq = self.__oCal.take_measurement(meastime)

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

            self.__oScX.close_port()
            self.__oScY.close_port()

        return True

    def start_CalEar(self, freqtx=433.9e6, freqspan=2e4):
        import rf
        self.__oCal = rf.CalEar(freqtx, freqspan)
        return True

    def start_LocEar(self, alpha, xi, txpos, freqtx, freqspan=2e4):
        import rf
        self.__oLoc = rf.LocEar(alpha, xi, txpos, freqtx, freqspan)
        return True


