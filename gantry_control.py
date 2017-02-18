import numpy as np
import matplotlib.pyplot as plt
import time as t
import serial_control as sc
import hippocampus_toolbox as hc_tools
import rf_tools


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
            # set home position knwon flao
            self.__oScX.set_home_pos_known(True)
            self.__oScY.set_home_pos_known(True)

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
            valid_wp = True
        else:
            print ('ERROR: Target way-point cannot be approached!')
            print ('Target way-point ' + str(wp) + ' does not lie within the gantry workspace ' +
                   'x= [' + str(gantry_dim[0]) + ' ... ' + str(gantry_dim[2]) + '], ' +
                   'y= [' + str(gantry_dim[1]) + ' ... ' + str(gantry_dim[3]) + '] ')
            valid_wp = False
        return valid_wp

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
    def follow_wp(self, start_wp, wp_list):


        num_wp = len(wp_list)
        print('Number of way points: ' + str(num_wp))
        start_time = t.time()

        self.set_target_wp(start_wp)
        self.start_moving_gantry_to_target()
        print('Moving to start position = ' + str(start_wp))
        while not self.confirm_arrived_at_target_wp():
            t.sleep(0.2)
        print('Arrived at start point')

        t.sleep(0.5)
        print('Start following way point sequence')


        meas_counter = 0
        time_elapsed = 0.0
        for wp in wp_list:
            # go to wp
            self.set_target_wp(wp)
            self.start_moving_gantry_to_target()
            not_arrived_at_wp = True
            print('Moving to wp = ' + str(wp))

            # moving to next wp
            while not_arrived_at_wp:
                meas_counter += 1
                time_elapsed = t.time() - start_time
                pos_x_mm, pos_y_mm = self.get_gantry_pos_xy_mm()





                if self.confirm_arrived_at_target_wp():
                    not_arrived_at_wp = False

            meas_freq = meas_counter / time_elapsed
            print('Logging with avg. ' + str(meas_freq) + ' Hz')



        return True



    def follow_wp_and_take_measurements(self, start_wp, wp_list, filename, set_sample_size=256):
        self.__oCal.set_size(set_sample_size)
        sample_size = self.__oCal.get_size()

        num_wp = len(wp_list)
        print('Number of way points: ' + str(num_wp))
        start_time = t.time()

        self.set_target_wp(start_wp)
        self.start_moving_gantry_to_target()
        print('Moving to start position = ' + str(start_wp))
        while not self.confirm_arrived_at_target_wp():
            t.sleep(0.2)
        print('Arrived at start point')

        t.sleep(0.5)
        print('Start following way point sequence')

        data_list = []
        meas_counter = 0
        time_elapsed = 0.0
        for wp in wp_list:
            # go to wp
            self.set_target_wp(wp)
            self.start_moving_gantry_to_target()
            not_arrived_at_wp = True
            print('Moving to wp = ' + str(wp))

            # taking measurements
            while not_arrived_at_wp:
                meas_counter += 1
                time_elapsed = t.time() - start_time
                pos_x_mm, pos_y_mm = self.get_gantry_pos_xy_mm()
                freq_den_max, pxx_den_max = self.__oCal.get_rss_peaks_from_single_sample()

                data_row = np.append([meas_counter, time_elapsed, pos_x_mm, pos_y_mm], pxx_den_max)
                data_list.append(data_row)

                if self.confirm_arrived_at_target_wp():
                    not_arrived_at_wp = False

            meas_freq = meas_counter / time_elapsed
            print('Logging with avg. ' + str(meas_freq) + ' Hz')

        with open(filename, 'w') as measfile:
            measfile.write('Measurement file for trajectory following\n')
            measfile.write('Measurement was taken on ' + t.ctime() + '\n')
            measfile.write('### begin grid settings\n')
            measfile.write('sample size = ' + str(sample_size) + ' [*1024]\n')
            measfile.write('avg. meas frequency = ' + str(meas_freq) + ' Hz\n')
            measfile.write('start_point =' + str(start_wp) + '\n')
            measfile.write('wp_list =' + str(wp_list) + '\n')
            measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm], pxx_den_max\n')
            measfile.write('### begin data log\n')
            data_mat = np.asarray(data_list)
            for row in data_mat:
                row_string = ''
                for i in range(len(row)):
                    row_string += str(row[i]) + ','
                row_string += '\n'
                measfile.write(row_string)

            measfile.close()

        return True

    def position_hold_measurements(self, xy_pos_mm, meas_time, filename, set_sample_size=256):
        """

        :param xy_pos_mm:
        :param meas_time:
        :param filename:
        :param set_sample_size:
        :return:
        """
        self.set_target_wp(xy_pos_mm)
        self.start_moving_gantry_to_target()
        while not self.confirm_arrived_at_target_wp():
            print('Moving to position = ' + str(xy_pos_mm))
            t.sleep(0.2)

        self.__oCal.set_size(set_sample_size)
        sample_size = self.__oCal.get_size()

        print('Sampling with sample size ' + str(sample_size) + ' [*1024]\n')

        start_time = t.time()
        print('measuring for ' + str(meas_time) + 's ...\n')
        time_elapsed = 0.0
        meas_counter = 0.0
        data_list = []

        # taking measurements
        while time_elapsed < meas_time:
            pos_x_mm, pos_y_mm = self.get_gantry_pos_xy_mm()
            freq_den_max, pxx_den_max = self.__oCal.get_rss_peaks_from_single_sample()
            time_elapsed = t.time() - start_time
            meas_counter += 1.0

            data_row = np.append([meas_counter, time_elapsed, pos_x_mm, pos_y_mm], pxx_den_max)
            data_list.append(data_row)

        meas_freq = meas_counter / time_elapsed
        print('Logging with avg. ' + str(meas_freq) + ' Hz')

        # save data to file
        with open(filename, 'w') as measfile:
            measfile.write('Measurement file for trajectory following\n')
            measfile.write('Measurement was taken on ' + t.ctime() + '\n')
            measfile.write('### begin grid settings\n')
            measfile.write('measurements at position = ' + str(xy_pos_mm) + '\n')
            measfile.write('Meas_time = ' + str(meas_time) + '\n')
            measfile.write('sample size = ' + str(sample_size) + ' [*1024]\n')
            measfile.write('avg. meas frequency = ' + str(meas_freq) + ' Hz\n')
            measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm], pxx_den_max\n')
            measfile.write('### begin data log\n')
            data_mat = np.asarray(data_list)
            for row in data_mat:
                row_string = ''
                for i in range(len(row)):
                    row_string += str(row[i]) + ','
                row_string += '\n'
                measfile.write(row_string)

            measfile.close()

        return True

    def process_measurement_sequence(self, wplist_filename, measdata_filename, numtx, tx_abs_pos, freqtx):
        """
        :return:
        """
        print('Process Measurement Sequence started')
        # read data from waypoint file
        #wplist_filename = hc_tools.select_file()

        """
        with open(wplist_filename, 'r') as wpfile:

            for i, line in enumerate(wplist_filename):
                print('i= ' + str(i) + ' line:' + line)
                if line == '###':
                if i >= 3:  # ignore header (first 3 lines)

            wp_data_list = [map(float, line.split(',')) for line in wpfile]
            wp_data_mat = np.asarray(wp_data_list)
            wpfile.close()
        """
        #wp_data_mat, x0, xn, grid_dxdy, timemeas = rf_tools.read_data_from_wp_list_file(wplist_filename)
        with open(wplist_filename, 'r') as wpfile:
            load_description = True
            load_grid_settings = False
            load_wplist = False
            wp_append_list = []
            for i, line in enumerate(wpfile):

                if line == '### begin grid settings\n':
                    print('griddata found')
                    load_description = False
                    load_grid_settings = True
                    load_wplist = False
                    continue
                elif line == '### begin wp_list\n':
                    load_description = False
                    load_grid_settings = False
                    load_wplist = True
                    print('### found')
                    continue
                if load_description:
                    print('file description')
                    print(line)

                if load_grid_settings and not load_wplist:
                    grid_settings = map(float, line.split(','))
                    x0 = [grid_settings[0], grid_settings[1]]
                    xn = [grid_settings[2], grid_settings[3]]
                    grid_dxdy = [grid_settings[4], grid_settings[5]]
                    timemeas = grid_settings[6]

                    data_shape = [xn[0] / grid_dxdy[0] + 1, xn[1] / grid_dxdy[1] + 1]

                if load_wplist and not load_grid_settings:
                    # print('read wplist')
                    wp_append_list.append(map(float, line.split(',')))

            print(str(np.asarray(wp_append_list)))
            wp_data_mat = np.asarray(wp_append_list)

            wpfile.close()

        #measdata_filename = hc_tools.save_as_dialog('Save measurement data as...')
        with open(measdata_filename, 'w') as measfile:

            # write header to measurement file
            file_description = 'Measurement file\n' + 'Measurement was taken on ' + t.ctime() + '\n'

            txdata = str(numtx) + ', '
            for itx in range(numtx):
                txpos = tx_abs_pos[itx]
                txdata += str(txpos[0]) + ', ' + str(txpos[1]) + ', '
            for itx in range(numtx):
                txdata += str(freqtx[itx]) + ', '

            print('txdata = ' + txdata)

            measfile.write(file_description)
            measfile.write('### begin grid settings\n')
            measfile.write(str(x0[0]) + ', ' + str(x0[1]) + ', ' +
                           str(xn[0]) + ', ' + str(xn[1]) + ', ' +
                           str(grid_dxdy[0]) + ', ' + str(grid_dxdy[1]) + ', ' +
                           str(timemeas) + ', ' + txdata +
                           '\n')
            measfile.write('### begin measurement data\n')

            # setup plot
            plt.ion()
            plt.plot(wp_data_mat[:, 1], wp_data_mat[:, 2], 'b.-')
            plt.xlabel('Distance in mm (belt-drive)')
            plt.ylabel('Distance in mm (spindle-drive)')
            plt.xlim(x0[0]-10, xn[0]+100)
            plt.ylim(x0[1]-10, xn[1]+100)
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


