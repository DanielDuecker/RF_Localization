import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time as t
import serial_control as sc
import hippocampus_toolbox as hc_tools
import rf_tools
import estimator
import sys


class GantryControl(object):
    def __init__(self, gantry_dimensions=[0, 3000, 0, 1580, 0, 500], use_gui=False, sdr_type='NooElec'):  # [x0 ,x1, y0, y1]
        self.__dimensions = gantry_dimensions
        self.__gantry_pos = [0, 0, 0]  # initial position after start
        self.__target_wp_mmrad = []
        self.__oRf = []
        # self.__oCal = []
        self.__oLoc = []
        self.__oScX = []  # spindle-drive
        self.__oScY = []  # belt-drive
        self.__oScZ = []  # threaded-rod-drive
        self.__maxposdeviation_mm = 2  # [mm] max position deviation per axis
        self.__maxposdeviation_rad = 10e-2  # [rad] max position deviation per axis

        self.__oScX = sc.MotorCommunication('/dev/ttyS0', 'belt_drive', 115200, 'belt', 3100, 2000e3)
        self.__oScY = sc.MotorCommunication('/dev/ttyS1', 'spindle_drive', 19200, 'spindle', 1600, 945800)
        self.__oScZ = sc.MotorCommunication('/dev/ttyUSB2', 'tread_drive', 19200, 'threadedrod', 700, 7590)  # USB0 / USB1 / USB2 ...

        self.__sdr_type = sdr_type

        self.__starttime = []

        if use_gui:
            print('Gantry Control - gui mode')
            self.__oScX.open_port()
            self.__oScY.open_port()
            self.__oScZ.open_port()

            # set home position known flao
            self.__oScX.set_home_pos_known(True)
            self.__oScY.set_home_pos_known(True)
            self.__oScZ.set_home_pos_known(True)

        else:

            # belt-drive
            self.__oScX.open_port()
            self.__oScX.start_manual_mode()
            self.__oScX.enter_manual_init_data()
            if self.__oScX.get_manual_init() is False:
                self.__oScX.initialize_home_pos()
                # self.__oScX.initialize_extreme_pos()
            print('Belt-Drive: Setup DONE!')

            # spindle-drive
            self.__oScY.open_port()
            self.__oScY.start_manual_mode()
            self.__oScY.enter_manual_init_data()
            if self.__oScY.get_manual_init is False:
                self.__oScY.initialize_home_pos()
                # self.__oScY.initialize_extreme_pos()
            print('Spindle-Drive: Setup DONE!')

            # rod-drive
            self.__oScZ.open_port()
            self.__oScZ.start_manual_mode()
            self.__oScZ.enter_manual_init_data()
            if self.__oScZ.get_manual_init is False:
                self.__oScZ.initialize_home_pos()
                # self.__oScZ.initialize_extreme_pos()
            print('Rod-Drive: Setup DONE!')

    def get_serial_x_handle(self):
        return self.__oScX

    def get_serial_y_handle(self):
        return self.__oScY

    def get_serial_z_handle(self):
        return self.__oScZ

    def get_gantry_dimensions(self):
        return self.__dimensions

    def get_gantry_pos(self):
        return self.__gantry_pos

    def set_starttime(self):
        self.__starttime = t.ctime()

    def get_starttime(self):
        return self.__starttime

    def set_new_max_speed_x(self, max_speed):
        belt_speed_limit = 3000
        if max_speed > belt_speed_limit:
            print('Warning: Not able to set max belt speed to '+str(max_speed)+' limit is ' + str(belt_speed_limit) + '!!!')
            return True
        self.__oScX.set_drive_max_speed(max_speed)
        print('Set new belt max speed to ' + str(max_speed))
        return True

    def set_new_max_speed_y(self, max_speed):
        spindle_speed_limit = 9000
        if max_speed > spindle_speed_limit:
            print('Warning: Not able to set max spindle speed to '+str(max_speed)+' limit is ' + str(spindle_speed_limit) + '!!!')
            return True
        self.__oScY.set_drive_max_speed(max_speed)
        print('Set new spindle max speed to ' + str(max_speed))
        return True

    def set_new_max_speed_z(self, max_speed):
        rod_speed_limit = 101
        if max_speed > rod_speed_limit:
            print('Warning: Not able to set max rod speed to '+str(max_speed)+' limit is ' + str(rod_speed_limit) + '!!!')
            return True
        self.__oScZ.set_drive_max_speed(max_speed)
        print('Set new rod max speed to ' + str(max_speed))
        return True

    def go_to_abs_pos(self, pos_x, pos_y, pos_a):
        self.__oScX.go_to_pos_mmrad(pos_x)
        self.__oScY.go_to_pos_mmrad(pos_y)
        self.__oScZ.go_to_pos_mmrad(pos_a)
        print('Move gantry to position x = ' + str(pos_x) + 'mm y = ' + str(pos_y) + 'mm alpha = ' + str(pos_a))
        return True

    def go_to_rel_pos(self, dx_pos, dy_pos, da_pos):
        self.__oScX.go_to_delta_pos_mmrad(dx_pos)
        self.__oScY.go_to_delta_pos_mmrad(dy_pos)
        self.__oScZ.go_to_delta_pos_mmrad(da_pos)
        print('Move gantry by  dx= ' + str(dx_pos) + 'mm dy = ' + str(dy_pos) + 'mm dalpha = ' + str(da_pos))
        return True

    def set_gantry_pos(self, new_pos):
        self.__gantry_pos = new_pos

    def set_target_wp(self, target_wp):
        if len(target_wp) == len(self.__gantry_pos):
            if self.check_wp_in_workspace(target_wp):
                self.__target_wp_mmrad = target_wp
                b_new_wp = True
            else:
                print('ERROR:target way-point: x=' + str(target_wp(0) + ' y=' + str(target_wp(1)) + ' alpha=' + str(target_wp(2)) + ' not in workspace'))
                b_new_wp = False
        else:
            print('ERROR: Dimension mismatch!')
            print('len(target_wp) ='+str(len(target_wp))+' ~= len(self.__gantry_pos)  ='+str(len(self.__gantry_pos)))
            b_new_wp = False
        return b_new_wp

    def get_target_wp_mmrad(self):
        return self.__target_wp_mmrad

    def get_gantry_pos_xyz_mmrad(self):
        pos_x_mm = self.__oScX.get_posmmrad()  # belt-drive
        pos_y_mm = self.__oScY.get_posmmrad()  # spindle-drive
        pos_z_mm = self.__oScZ.get_posmmrad()  # rod-drive
        return pos_x_mm, pos_y_mm, pos_z_mm

    def start_go_home_seq_xyz(self):
        self.__oScX.start_home_seq()  # belt-drive
        self.__oScY.start_home_seq()  # spindle-drive

    def check_wp_in_workspace(self, wp):
        gantry_dim = self.get_gantry_dimensions()

        if wp[0] >= gantry_dim[0] and wp[0] <= gantry_dim[1] and wp[1] >= gantry_dim[2] and wp[1] <= gantry_dim[3] and wp[2] >= gantry_dim[4] and wp[2] <= gantry_dim[5]:
            valid_wp = True
        else:
            print ('ERROR: Target way-point cannot be approached!')
            print ('Target way-point ' + str(wp) + ' does not lie within the gantry workspace ' +
                   'x= [' + str(gantry_dim[0]) + ' ... ' + str(gantry_dim[1]) + '], ' +
                   'y= [' + str(gantry_dim[2]) + ' ... ' + str(gantry_dim[3]) + '], ' +
                   'alpha= [' + str(gantry_dim[4]) + ' ... ' + str(gantry_dim[5]) + ']')
            valid_wp = False
        return valid_wp

    def transmit_wp_to_gantry(self, targetwp):
        if self.set_target_wp(targetwp):
            btransmission = True
        else:
            print ('ERROR: wp-transmission to gantry failed!')
            btransmission = False
        return btransmission

    def confirm_arrived_at_target_wp(self, tolx_mm=2, toly_mm=2, tolz_mm=1):
        """
        This method checks whether the gantry has arrived at its target position
        within a range of 'maxdeviation' [mm]
        :param tolx: position tolerance
        :param toly: position tolerance
        :return:
        """
        """

        :return: flag - arrived true/false
        """
        barrival_confirmed = False
        gantry_pos_mmrad = self.get_gantry_pos_xyz_mmrad()
        target_pos_mmrad = self.get_target_wp_mmrad()
        distx = abs(gantry_pos_mmrad[0] - target_pos_mmrad[0])
        disty = abs(gantry_pos_mmrad[1] - target_pos_mmrad[1])
        dista = abs(gantry_pos_mmrad[2] - target_pos_mmrad[2])

        if distx < tolx_mm and disty < toly_mm and dista < tolz_mm:
            barrival_confirmed = True

        return barrival_confirmed

    def start_moving_gantry_to_target(self):
        target_wp = self.get_target_wp_mmrad()

        print ('Move gantry to way-point x [mm] = ' + str(target_wp[0]) + ' y [mm] = ' + str(target_wp[1]) + ' z [mm] = ' + str(target_wp[2]))
        self.__oScX.go_to_pos_mmrad(target_wp[0])
        self.__oScY.go_to_pos_mmrad(target_wp[1])
        self.__oScZ.go_to_pos_mmrad(target_wp[2])

    def move_gantry_to_target(self):
        self.start_moving_gantry_to_target()

        bArrived_all = False
        while bArrived_all is False:
            t.sleep(0.01)

            actpos_X_mm, actpos_Y_mm, actpos_Z_mm = self.get_gantry_pos_xyz_mmrad()
            actpos = [actpos_X_mm, actpos_Y_mm, actpos_Z_mm]
            # print('Actual position: x = ' + str(round(actpos[0], 1)) + 'mm y = ' + str(round(actpos[1], 1)) + 'mm z = ' + str(round(actpos[2], 1)) + 'mm')
            self.set_gantry_pos(actpos)
            """
            dist_x = abs(self.get_gantry_pos()[0] - target_wp[0])
            dist_y = abs(self.get_gantry_pos()[1] - target_wp[1])
            dist_z = abs(self.get_gantry_pos()[2] - target_wp[2])
            if dist_x < self.__maxposdeviation_mm and dist_y < self.__maxposdeviation_mm and dist_a < self.__maxposdeviation_mm:
                print ('Arrived at way-point')
                bArrived_all = True
            """
            if self.confirm_arrived_at_target_wp():
                print ('Arrived at way-point')
                bArrived_all = True

        # @todo: position feedback from motor  check whether position is within a suitable range

        return bArrived_all

    def move_gantry_to_target_manual(self):
        target_wp = self.get_target_wp_mmrad()

        print ('move gantry to way-point x [mm] = ' + str(target_wp[0]) + ' y [mm] = ' + str(target_wp[1]) + ' z [mm] = ' + str(target_wp[2]))

        # some control stuff can be inserted here
        t.sleep(0.4)
        self.set_gantry_pos(target_wp)
        if self.get_gantry_pos() == target_wp:
            print ('arrived at new waypoint')
            inc_pos = target_wp[0] * 1e6 / 310 + 1e6  # Probably false for [rad]-motors
            raw_input(' confirm arrival at INC pos x [inc] = ' + str(inc_pos))
            bArrived = True
        else:
            print ('Gantry haven t arrived at target way-point')
            actual_wp = self.get_gantry_pos()
            print ('Actual position: x= ' + str(actual_wp[0]) + ' y= ' + str(actual_wp[1]) + ' alpha= ' + str(actual_wp[2]) +
                   ' target wp x= ' + str(target_wp[0]) + ' y= ' + str(target_wp[1]) + ' alpha= ' + str(target_wp[2]))
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
                pos_x_mm, pos_y_mm, pos_z_mm = self.get_gantry_pos_xyz_mmrad()
                if self.confirm_arrived_at_target_wp():
                    not_arrived_at_wp = False

            meas_freq = meas_counter / time_elapsed
            print('Logging with avg. ' + str(meas_freq) + ' Hz')
        return True

    def follow_wp_and_take_measurements(self, start_wp=[1000, 1000, 100], sample_size=32):

        self.start_RfEar()
        self.__oRf.set_samplesize(sample_size)
        sample_size = self.__oRf.get_samplesize()

        wplist_filename = hc_tools.select_file()
        print(wplist_filename)
        wp_append_list = []
        with open(wplist_filename, 'r') as wpfile:
            for i, line in enumerate(wpfile):
                print('line = ' + line)
                print(line[:-2].split(' '))
                temp_list = line[:-2].split(' ')
                wp_append_list.append(map(float, temp_list[0:-1]))

        print(str(np.asarray(wp_append_list)))
        wp_list = np.asarray(wp_append_list)

        measdata_filename = hc_tools.save_as_dialog()
        print(measdata_filename)

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
                pos_x_mm, pos_y_mm, pos_z_mm = self.get_gantry_pos_xyz_mmrad()
                freq_den_max, pxx_den_max = self.__oRf.get_rss_peaks()

                data_row = np.append([meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max)
                data_list.append(data_row)

                if self.confirm_arrived_at_target_wp():
                    not_arrived_at_wp = False

            meas_freq = meas_counter / time_elapsed
            print('Logging with avg. ' + str(meas_freq) + ' Hz')

        with open(measdata_filename, 'w') as measfile:
            measfile.write('Measurement file for trajectory following\n')
            measfile.write('Measurement was taken on ' + t.ctime() + '\n')
            measfile.write('### begin grid settings\n')
            measfile.write('sample size = ' + str(sample_size) + ' [*1024]\n')
            measfile.write('avg. meas frequency = ' + str(meas_freq) + ' Hz\n')
            measfile.write('start_point =' + str(start_wp) + '\n')
            measfile.write('wp_list =' + str(wp_list) + '\n')
            measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max\n')
            measfile.write('### begin data log\n')
            data_mat = np.asarray(data_list)
            for row in data_mat:
                row_string = ''
                for i in range(len(row)):
                    row_string += str(row[i]) + ' '
                row_string += '\n'
                measfile.write(row_string)

            measfile.close()

        return True

    def follow_wp_path_opt_take_measurements(self, num_plot_points=250, model_type='log', b_take_meas=False, b_log_data=False, tolmmbroad=10, tolmmfine= 1, start_wp=[1000, 1000, 100], sample_size=32):
        """
        :param b_take_meas:
        :param start_wp:
        :param sample_size:
        :return:
        """
        if b_take_meas is True:  # dont start SDR if not RSS measurements will be taken
            self.start_RfEar()
            self.__oRf.set_samplesize(sample_size)
            sample_size = self.__oRf.get_samplesize()

        wplist_filename = hc_tools.select_file()
        print(wplist_filename)
        wp_append_list = []
        with open(wplist_filename, 'r') as wpfile:
            for i, line in enumerate(wpfile):
                # print('line = ' + line)
                # print(line[:-2].split(' '))
                temp_list = line[:-2].split(' ')
                wp_append_list.append(map(float, temp_list[0:-1]))

#        print(str(np.asarray(wp_append_list)))
        wp_list = np.asarray(wp_append_list)
        if b_log_data:
            measdata_filename = hc_tools.save_as_dialog()
 #       print(measdata_filename)

        num_wp = len(wp_list)
        print('Number of way points: ' + str(num_wp))
        start_time = t.time()

        data_list = []
        meas_counter = 0
        time_elapsed = 0.0
        tolx_mm = tolmmbroad  # mm
        toly_mm = tolmmbroad  # mm
        tolz_mm = tolmmfine  # mm

        b_ekf = True
        if b_ekf is True:
            # init EKF
            EKF = estimator.ExtendedKalmanFilter(model_type)
            import estimator_plot_tools
            EKF_plotter = estimator_plot_tools.EKF_Plot(EKF.get_tx_pos(), model_type)

        # follow wp sequence
        for wp in wp_list:
            print('wp in list = ' + str(wp))
            # go to wp
            self.set_target_wp(wp)
            self.start_moving_gantry_to_target()
            not_arrived_at_wp = True
            print('Moving to wp = ' + str(wp))

            # following sequence
            while not_arrived_at_wp:
                meas_counter += 1
                time_elapsed = t.time() - start_time
                pos_x_mm, pos_y_mm, pos_z_mm = self.get_gantry_pos_xyz_mmrad()

                if b_ekf is True:
                    EKF.ekf_prediction()
                    EKF.ekf_update(-85)
                    # EKF.check_valid_position_estimate()
                    # print(EKF.get_x_est())
                    EKF_plotter.add_x_est_to_plot(EKF.get_x_est())
                    EKF_plotter.update_meas_circles(EKF.get_z_meas(), EKF.get_tx_alpha(), EKF.get_tx_gamma(), True, EKF.get_y_est())
                    EKF_plotter.plot_gantry_pos([pos_x_mm, pos_y_mm])
                    EKF_plotter.plot_ekf_pos_live(True, num_plot_points)
                    # EKF_plotter.add_p_cov_to_plot(EKF.get_p_mat())
                    #EKF_plotter.plot_p_cov(num_plot_points)


                if b_take_meas is True:
                    # taking measurements
                    freq_den_max, pxx_den_max = self.__oRf.get_rss_peaks()
                    data_row = np.append([meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max)

                else:
                    data_row = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm]

                # add new data to list
                data_list.append(data_row)

                # arrived at wp? -> go to next
                if self.confirm_arrived_at_target_wp(tolx_mm, toly_mm, tolz_mm):
                    not_arrived_at_wp = False

            meas_freq = meas_counter / time_elapsed
            print('Logging with avg. ' + str(meas_freq) + ' Hz')

        if b_log_data:
            with open(measdata_filename, 'w') as measfile:
                measfile.write('Measurement file for trajectory following\n')
                measfile.write('Measurement was taken on ' + t.ctime() + '\n')
                measfile.write('### begin grid settings\n')
                if b_take_meas is True:
                    measfile.write('sample size = ' + str(sample_size) + ' [*1024]\n')
                    measfile.write('avg. meas frequency = ' + str(meas_freq) + ' Hz\n')
                measfile.write('start_point =' + str(start_wp) + '\n')
                measfile.write('wp_list =' + str(wp_list) + '\n')
                if b_take_meas is True:
                    measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max\n')
                else:
                    measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm]\n')

                measfile.write('### begin data log\n')

                data_mat = np.asarray(data_list)
                for row in data_mat:
                    row_string = ''
                    for i in range(len(row)):
                        row_string += str(row[i]) + ' '
                    row_string += '\n'
                    measfile.write(row_string)

                measfile.close()

        return True

    def position_hold_measurements(self, xyz_pos_mmrad, meas_time, filename, set_sample_size=256):
        """

        :param xyz_pos_mmrad:
        :param meas_time:
        :param filename:
        :param set_sample_size:
        :return:
        """
        self.set_target_wp(xyz_pos_mmrad)
        self.start_moving_gantry_to_target()
        while not self.confirm_arrived_at_target_wp():
            print('Moving to position = ' + str(xyz_pos_mmrad))
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
            pos_x_mm, pos_y_mm, pos_z_mm = self.get_gantry_pos_xyz_mmrad()
            freq_den_max, pxx_den_max = self.__oCal.get_rss_peaks_at_freqtx()
            time_elapsed = t.time() - start_time
            meas_counter += 1.0

            data_row = np.append([meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max)
            data_list.append(data_row)

        meas_freq = meas_counter / time_elapsed
        print('Logging with avg. ' + str(meas_freq) + ' Hz')

        # save data to file
        with open(filename, 'w') as measfile:
            measfile.write('Measurement file for trajectory following\n')
            measfile.write('Measurement was taken on ' + t.ctime() + '\n')
            measfile.write('### begin grid settings\n')
            measfile.write('measurements at position = ' + str(xyz_pos_mmrad) + '\n')
            measfile.write('Meas_time = ' + str(meas_time) + '\n')
            measfile.write('sample size = ' + str(sample_size) + ' [*1024]\n')
            measfile.write('avg. meas frequency = ' + str(meas_freq) + ' Hz\n')
            measfile.write('data format = [meas_counter, time_elapsed, pos_x_mm, pos_y_mm, pos_z_mm], pxx_den_max\n')
            measfile.write('### begin data log\n')
            data_mat = np.asarray(data_list)
            for row in data_mat:
                row_string = ''
                for i in range(len(row)):
                    row_string += str(row[i]) + ' '
                row_string += '\n'
                measfile.write(row_string)

            measfile.close()

        return True

    def start_field_measurement_file_select(self):
        # read data from waypoint file

        wplist_filename = hc_tools.select_file()
        print(wplist_filename)

        measdata_filename = hc_tools.save_as_dialog()
        print(measdata_filename)

        meas_description = hc_tools.write_descrition()

        self.start_RfEar()
        freqtx, numtx, tx_abs_pos = self.__oRf.get_txparams()
        print(freqtx)
        print(numtx)
        print(tx_abs_pos)

        self.process_measurement_sequence(wplist_filename, measdata_filename, numtx, tx_abs_pos, freqtx, meas_description)

    def process_measurement_sequence(self, wplist_filename, measdata_filename, numtx, tx_abs_pos, freqtx, meas_description):
        """
        :return:
        """
        print('Process Measurement Sequence started')
        # read data from waypoint file
        # wplist_filename = hc_tools.select_file()

        """
        with open(wplist_filename, 'r') as wpfile:

            for i, line in enumerate(wplist_filename):
                print('i= ' + str(i) + ' line:' + line)
                if line == '###':
                if i >= 3:  # ignore header (first 3 lines)

            wp_data_list = [map(float, line[:-2].split(' ')) for line in wpfile]
            wp_data_mat = np.asarray(wp_data_list)
            wpfile.close()
        """
        # wp_data_mat, x0, xn, grid_dxdy, timemeas = rf_tools.read_data_from_wp_list_file(wplist_filename)
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
                    grid_settings = map(float, line.split(' '))
                    x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                    xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                    grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                    timemeas = grid_settings[9]

                    data_shape = []
                    for i in range(3):  # range(num_dof)
                        try:
                            shapei = int((xn[i] - x0[i]) / grid_dxdyda[i] + i)
                        except ZeroDivisionError:
                            shapei = 1
                        data_shape.append(shapei)

                    # old: data_shape = [xn[0] / grid_dxdyda[0] + 1, xn[1] / grid_dxdyda[1] + 1, xn[2] / grid_dxdyda[2] + 1]

                if load_wplist and not load_grid_settings:
                    # print('read wplist')
                    wp_append_list.append(map(float, line[:-2].split(' ')))

            wp_data_mat = np.asarray(wp_append_list)
            print(str(wp_data_mat))

            wpfile.close()

        #measdata_filename = hc_tools.save_as_dialog('Save measurement data as...')
        with open(measdata_filename, 'w') as measfile:

            # write header to measurement file
            file_description = 'Measurement file\n' + 'Measurement was taken on ' + t.ctime() + '\n' + 'Description: ' + meas_description + '\n'

            txdata = str(numtx) + ' '
            for itx in range(numtx):
                txpos = tx_abs_pos[itx]
                txdata += str(txpos[0]) + ' ' + str(txpos[1]) + ' ' + str(txpos[2]) + ' '
            for itx in range(numtx):
                txdata += str(freqtx[itx]) + ' '

            print('txdata = ' + txdata)

            measfile.write(file_description)
            measfile.write('### begin grid settings\n')
            measfile.write(str(x0[0]) + ' ' + str(x0[1]) + ' ' + str(x0[2]) + ' ' +
                           str(xn[0]) + ' ' + str(xn[1]) + ' ' + str(xn[2]) + ' ' +
                           str(grid_dxdyda[0]) + ' ' + str(grid_dxdyda[1]) + ' ' + str(grid_dxdyda[2]) + ' ' +
                           str(timemeas) + ' ' + txdata +
                           '\n')
            measfile.write('### begin measurement data\n')

            # setup plot
            fig = plt.figure()
            # plt.ion()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(wp_data_mat[:, 1], wp_data_mat[:, 2], wp_data_mat[:, 3], 'b.-')
            ax.set_xlabel('Distance in mm (belt-drive)')
            ax.set_ylabel('Distance in mm (spindle-drive)')
            ax.set_zlabel('Distance in mm (rod-drive)')
            ax.set_xlim(-100, 3100)
            ax.set_ylim(-100, 1800)
            ax.set_zlim(-10, 1000)
            # plt.xlim(x0[0]-10, xn[0]+100)  # bei Bedarf mit ax.___
            # plt.ylim(x0[1]-10, xn[1]+100)  # bei Bedarf mit ax.___
            # plt.grid()
            for i in range(len(tx_abs_pos)):
                txpos_single = tx_abs_pos[i]
                ax.scatter(txpos_single[0], txpos_single[1],  txpos_single[2], c='red')
            ax.draw

            totnumofwp = np.shape(wp_data_mat)

            totnumofwp = totnumofwp[0]
            print ('Number of waypoints = ' + str(totnumofwp) + '\n')

<<<<<<< HEAD
            z_meas = [300.0]
            import Tkinter
            import tkMessageBox

            for z in z_meas:
                tkMessageBox.showinfo("Information", "Schicht ist fertig gemessen. Bitte Z-Achse auf " + str(z) + "mm stellen und bestaetigen...")

                # loop over all way-points
                for row in wp_data_mat:

                    numwp = int(row[0])
                    new_target_wpx = row[1]
                    new_target_wpy = row[2]
                    new_target_wpz = row[3]
                    new_target_wp = [new_target_wpx, new_target_wpy, new_target_wpz]  # find a solution for this ugly workaround...
                    meastime = row[4]

                    # estimate time left for plot title
                    if numwp == 0:
                        starttime = float(t.time())
                        t_left_h = 0
                        t_left_m = 0
                        t_left_s = 0
                    else:
                        time_per_point = (float(t.time()) - starttime) / (numwp + 1)  # as numwp starts at 0
                        time_left_sec = time_per_point * (totnumofwp-numwp+1)
                        m, t_left_s = divmod(time_left_sec, 60)
                        t_left_h, t_left_m = divmod(m, 60)

                    if self.transmit_wp_to_gantry(new_target_wp):
                        if self.move_gantry_to_target():
                            if self.confirm_arrived_at_target_wp():
                                t.sleep(.25)  # wait to damp motion/oscillation of antenna etc

                                print('START Measurement for ' + str(meastime) + 's')
                                print('Measuring at Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points')
                                ax.scatter(new_target_wp[0], new_target_wp[1], zs=new_target_wp[2], c='gold')
                                temp_meas_title = 'Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points ' +'- Time left: %d:%02d:%02d' % (t_left_h, t_left_m, t_left_s)
                                ax.set_title(temp_meas_title, loc='left')
                                # dataseq = self.__oCal.take_measurement(meastime)
                                dataseq = self.__oRf.take_measurement(meastime)

                                [nummeas, numtx] = np.shape(dataseq)

                                # way point data - structure 'wp_x, wp_y, wp_a, num_wp, num_tx, num_meas'
                                # str_base_data = str(new_target_wp[0]) + ' ' + str(new_target_wp[1]) + ' ' + str(new_target_wp[2]) + ' ' + str(numwp) + ' ' + str(numtx) + ' ' + str(nummeas) + ' '

                                str_base_data = str(new_target_wp[0]) + ' ' + str(new_target_wp[1]) + ' ' + str(z) + ' ' + str(numwp) + ' ' + str(numtx) + ' ' + str(nummeas) + ' '

                                # freq data
                                str_freqs = ' '.join(map(str, freqtx)) + ' '

                                # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
                                # print('data ' + str(dataseq))
                                str_rss = ''
                                #print(dataseq)
                                for i in range(numtx):
                                    str_rss = str_rss + ' '.join(map(str, np.matrix.round(dataseq[:, i], decimals=3))) + ' '

                                measfile.write(str_base_data + str_freqs + str_rss + '\n')
                                # print(str_base_data + str_freqs + str_rss)

                        else:
                            print ('Error: Failed to move gantry to new way-point!')
                            print ('Way-point #' + str(numwp) + ' @ position x= ' + str(new_target_wp[0]) + ', y= '
                                   + str(new_target_wp[1])) + ' @ position z= ' + str(new_target_wp[2])
=======
            # loop over all way-points
            for row in wp_data_mat:

                numwp = int(row[0])
                new_target_wpx = row[1]
                new_target_wpy = row[2]
                new_target_wpz = row[3]
                new_target_wp = [new_target_wpx, new_target_wpy, new_target_wpz]  # find a solution for this ugly workaround...
                meastime = row[4]

                # estimate time left for plot title
                if numwp == 0:
                    starttime = float(t.time())
                    t_left_h = 0
                    t_left_m = 0
                    t_left_s = 0
                else:
                    time_per_point = (float(t.time()) - starttime) / (numwp + 1)  # as numwp starts at 0
                    time_left_sec = time_per_point * (totnumofwp-numwp+1)
                    m, t_left_s = divmod(time_left_sec, 60)
                    t_left_h, t_left_m = divmod(m, 60)

                if self.transmit_wp_to_gantry(new_target_wp):
                    if self.move_gantry_to_target():
                        if self.confirm_arrived_at_target_wp():
                            t.sleep(.25)  # wait to damp motion/oscillation of antenna etc

                            print('START Measurement for ' + str(meastime) + 's')
                            print('Measuring at Way-Point #' + str(numwp+1) + ' of ' + str(totnumofwp) + ' way-points')
                            ax.scatter(new_target_wp[0], new_target_wp[1], zs=new_target_wp[2], c='gold')
                            temp_meas_title = 'Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points ' +'- Time left: %d:%02d:%02d' % (t_left_h, t_left_m, t_left_s)
                            ax.set_title(temp_meas_title, loc='left')
                            # dataseq = self.__oCal.take_measurement(meastime)
                            dataseq = self.__oRf.take_measurement(meastime)

                            [nummeas, numtx] = np.shape(dataseq)

                            # way point data - structure 'wp_x, wp_y, wp_a, num_wp, num_tx, num_meas'
                            str_base_data = str(new_target_wp[0]) + ' ' + str(new_target_wp[1]) + ' ' + str(new_target_wp[2]) + ' ' + str(numwp) + ' ' + str(numtx) + ' ' + str(nummeas) + ' '
                            # freq data
                            str_freqs = ' '.join(map(str, freqtx)) + ' '

                            # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
                            # print('data ' + str(dataseq))
                            str_rss = ''
                            #print(dataseq)
                            for i in range(numtx):
                                str_rss = str_rss + ' '.join(map(str, np.matrix.round(dataseq[:, i], decimals=3))) + ' '

                            measfile.write(str_base_data + str_freqs + str_rss + '\n')
                            # print(str_base_data + str_freqs + str_rss)
>>>>>>> b2b3201f4641aa89192bce3d513da0f463306ce0

                    else:
                        print ('Error: Failed to transmit new way-point to gantry!')
                        print ('Way-point #' + str(numwp) + ' @ position x= ' + str(new_target_wp[0]) + ', y= '
                               + str(new_target_wp[1])) + ' @ position z= ' + str(new_target_wp[2])
                    plt.pause(0.001)
                    print
            measfile.close()

            self.__oScX.close_port()
            self.__oScY.close_port()
            self.__oScZ.close_port()

        return True

    def start_RfEar(self, center_freq=434.2e6, freqspan=1e5):
        import rf

        self.__oRf = rf.RfEar(self.__sdr_type, center_freq, freqspan)

        freq6tx = [434.325e6, 433.89e6, 434.475e6, 434.025e6, 434.62e6, 434.175e6]

        tx_6pos = [[770, 432, 0],
                   [1794, 437, 0],
                   [2814, 447, 0],
                   [2824, 1232, 0],
                   [1789, 1237, 0],
                   [774, 1227, 0]]

        self.__oRf.set_txparams(freq6tx, tx_6pos)
        return True


