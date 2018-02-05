import rf
import numpy as np
import time as t
import rf_tools


def manual_measurement_sequence(wplist_filename, measdata_filename, numtx, tx_abs_pos, freqtx):
    """
    :return:
    """
    print('Process Measurement Sequence started')

    with open(wplist_filename, 'r') as wpfile:
        load_description = True
        load_grid_settings = False
        load_wplist = False
        wp_append_list = []
        for i, line in enumerate(wpfile):

            if line == '### begin grid settings\n':
                # print('griddata found')
                load_description = False
                load_grid_settings = True
                load_wplist = False
                continue
            elif line == '### begin wp_list\n':
                load_description = False
                load_grid_settings = False
                load_wplist = True
                # print('### found')
                continue
            if load_description:
                # print('file description')
                print(line)

            if load_grid_settings and not load_wplist:
                grid_settings = map(float, line.split(','))
                x0 = [grid_settings[0], grid_settings[1]]
                xn = [grid_settings[2], grid_settings[3]]
                grid_dxdy = [grid_settings[4], grid_settings[5]]
                timemeas = grid_settings[6]

                #data_shape = [xn[0] / grid_dxdy[0] + 1, xn[1] / grid_dxdy[1] + 1]

            if load_wplist and not load_grid_settings:
                # print(line.split(','))
                wp_append_list.append(map(float, line.split(',')))

        print(str(np.asarray(wp_append_list)))
        wp_data_mat = np.asarray(wp_append_list)

        wpfile.close()

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

            raw_input('Type anything and press >ENTER< to start measurement at waypoint ' + str(new_target_wp))

            print('START Measurement for ' + str(meastime) + 's')
            print('Measuring at Way-Point #' + str(numwp) + ' of ' + str(totnumofwp) + ' way-points')

            dataseq = oRf.take_measurement(meastime)

            [nummeas, numtx] = np.shape(dataseq)

            # way point data - structure 'wp_x, wp_y, num_wp, num_tx, num_meas'
            str_base_data = str(new_target_wp[0]) + ', ' + str(new_target_wp[1]) + ', ' + \
                            str(numwp) + ', ' + str(numtx) + ', ' + str(nummeas) + ', '
            # freq data
            str_freqs = ', '.join(map(str, freqtx)) + ', '

            # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
            # print('data ' + str(dataseq))
            str_rss = ''
            for i in range(numtx):
                str_rss = str_rss + ', '.join(map(str, dataseq[:, i])) + ', '

            measfile.write(str_base_data + str_freqs + str_rss + '\n')

        measfile.close()

    return True



wplist_filename = 'way_points_cal_hippo_onboard.txt'
print(wplist_filename)

measdata_filename = 'measdata_onboard_cal_data.txt'
print(measdata_filename)

oRf = rf.RfEar(center_freq=434.2e6, freqspan=1e5)

freq6tx = [434.00e6,  434.15e6, 434.30e6, 434.45e6, 434.65e6, 433.90e6]

tx_6pos = [[520, 430],
           [1540, 430],
           [2570, 430],
           [2570, 1230],
           [1540, 1230],
           [530, 1230]]
oRf.set_txparams(freq6tx, tx_6pos)


freqtx, numtx, tx_abs_pos = oRf.get_txparams()

#manual_measurement_sequence(wplist_filename, measdata_filename, numtx, tx_abs_pos, freqtx)

rf_tools.analyze_measdata_from_file(b_onboard=True, measfilename=measdata_filename)
