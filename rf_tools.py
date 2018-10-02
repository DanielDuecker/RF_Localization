import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.special import lambertw
import socket

import hippocampus_toolbox as hc_tools
"""
independent methods related to the gantry
"""


def wp_generator(wp_filename, x0=[0, 0, 0], xn=[1200, 1200, 0], grid_dxdyda=[50, 50, 0], timemeas=12.0, show_plot=False):
    """
    :param wp_filename:
    :param x0: [x0,y0] - start position of the grid
    :param xn: [xn,yn] - end position of the grid
    :param steps: [numX, numY] - step size
    :param timemeas: - time [s] to wait at each position for measurements
    :return: wp_mat [x, y, t]
    """
    steps = []
    for i in range(3):  # range(num_dof)
        try:
            stepsi = (xn[i]-x0[i])/grid_dxdyda[i]+1
        except ZeroDivisionError:
            stepsi = 1
        steps.append(stepsi)
    # old: steps = [(xn[0]-x0[0])/grid_dxdyda[0]+1, (xn[1]-x0[1])/grid_dxdyda[1]+1, (xn[2]-x0[2])/grid_dxdyda[2]+1]
    print('wp-grid_shape = ' + str(steps))

    startx = x0[0]
    endx = xn[0]
    stepx = steps[0]

    starty = x0[1]
    endy = xn[1]
    stepy = steps[1]

    startz = x0[2]
    endz = xn[2]
    stepz = steps[2]

    xpos = np.linspace(startx, endx, stepx)
    ypos = np.linspace(starty, endy, stepy)
    zpos = np.linspace(startz, endz, stepz)

    wp_maty, wp_matz, wp_matx = np.meshgrid(ypos, zpos, xpos)  # put least moving axis second, then first, then last
    wp_vecx = np.reshape(wp_matx, (len(xpos)*len(ypos)*len(zpos), 1))
    wp_vecy = np.reshape(wp_maty, (len(ypos)*len(zpos)*len(xpos), 1))
    wp_vecz = np.reshape(wp_matz, (len(zpos)*len(xpos)*len(ypos), 1))
    wp_time = np.ones((len(xpos)*len(ypos)*len(zpos), 1)) * timemeas

    wp_mat = np.append(wp_vecx, np.append(wp_vecy, wp_vecz, axis=1), axis=1)
    wp_mat = np.append(wp_mat, wp_time, axis=1)

    # wp_filename = hc_tools.save_as_dialog('Save way point list as...')
    with open(wp_filename, 'w') as wpfile:
        wpfile.write('Way point list \n')
        wpfile.write('### begin grid settings\n')
        wpfile.write(str(x0[0]) + ' ' + str(x0[1]) + ' ' + str(x0[2]) + ' ' +
                     str(xn[0]) + ' ' + str(xn[1]) + ' ' + str(xn[2]) + ' ' +
                     str(grid_dxdyda[0]) + ' ' + str(grid_dxdyda[1]) + ' ' + str(grid_dxdyda[2]) + ' ' +
                     str(timemeas) +
                     '\n')

        wpfile.write('### begin wp_list\n')
        for i in range(wp_mat.shape[0]):
            wpfile.write(str(i) + ' ' + str(wp_mat[i, 0]) + ' ' + str(wp_mat[i, 1]) + ' ' + str(wp_mat[i, 2]) + ' '  + str(wp_mat[i, 3]) + '\n')
        wpfile.close()
    if show_plot:
        fig = plt.figure()
        # plt.ion()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(wp_mat[:, 0], wp_mat[:, 1], wp_mat[:, 2], '.-')
        # ax.show()
    print('Way point generator terminated!')
    return wp_filename  # file output [line#, x, y, a, time]


def read_data_from_wp_list_file(filename):
    with open(filename, 'r') as wpfile:
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
                grid_settings = map(float, line[:-2].split(' '))
                x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                timemeas = grid_settings[9]

                data_shape = [xn[0]/grid_dxdyda[0]+1, xn[1]/grid_dxdyda[1]+1, xn[2]/grid_dxdyda[2]+1]
                print('wp-grid_shape = ' + str(data_shape))

            if load_wplist and not load_grid_settings:
                # print('read wplist')
                wp_append_list.append(map(float, line[:-2].split(' ')))

        print(str(np.asarray(wp_append_list)))
        wp_data_mat = np.asarray(wp_append_list)

        wpfile.close()
        return wp_data_mat, x0, xn, grid_dxdyda, timemeas, data_shape


def write_measfile_header(ofile, file_description, x0, xn, grid_dxdyda, timemeas, numtx, tx_abs_pos, freqtx):
    txdata = str(numtx) + ' '
    for itx in range(numtx):
        txpos = tx_abs_pos[itx]
        txdata += str(txpos[0]) + ' ' + str(txpos[1]) + ' ' + str(txpos[2]) + ' '
    for itx in range(numtx):
        txdata += str(freqtx[itx]) + ' '
    # -> numtx, x1,y1,z2, x2,y2,z2, x3,y3,z3, x4,y4,z4, x5,y5,z5, x6,y6,z6, freq1, freq2, freq3, freq4, freq5, freq6
    #      0     1                      -                               18   19                   -              24

    print('txdata = ' + txdata)

    ofile.write('Way point list \n')
    ofile.write(file_description)
    ofile.write('### begin grid settings\n')
    ofile.write(str(x0[0]) + ' ' + str(x0[1]) + ' ' + str(x0[2]) + ' ' +
                str(xn[0]) + ' ' + str(xn[1]) + ' ' + str(xn[2]) + ' ' +
                str(grid_dxdyda[0]) + ' ' + str(grid_dxdyda[1]) + ' ' + str(grid_dxdyda[2]) + ' ' +
                str(timemeas) + ' ' + txdata + '\n')
    return True


# Vektor wird auf Ebene projeziert und Winkel mit main-Vektor gebildet
def get_angle_v_on_plane(v_x, v_1main, v_2):
    v_x_proj = np.dot(v_x.T, v_2)[0][0]*v_2 + np.dot(v_x.T, v_1main)[0][0]*v_1main
    if np.linalg.norm(v_x_proj) == 0:
        angle_x = np.pi*0.5
    elif (np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main))) > 1:
        angle_x = np.arccos((np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(
            v_1main))) - 1e-10)  # -1e-10, da PC gerne etwas mehr als 1 ausrechnet und daher arccos nicht funktioniert.
    else:
        angle_x = np.arccos(np.dot(v_x_proj.T, v_1main)[0][0] / (np.linalg.norm(v_x_proj) * np.linalg.norm(v_1main)))
    return angle_x


def get_angles(x_current_anglecalc, tx_pos_anglecalc, h_tx_anglecalc, z_mauv_anglecalc, h_mauv_anglecalc):
    dh_anglecalc = h_mauv_anglecalc - h_tx_anglecalc
    r_anglecalc = x_current_anglecalc - tx_pos_anglecalc
    r_abs_anglecalc = np.linalg.norm(r_anglecalc)
    phi_cap_anglecalc = np.arccos(r_anglecalc[0][0]/r_abs_anglecalc)
    if r_anglecalc[1][0] <= 0.0:
        phi_cap_anglecalc = 2*np.pi - phi_cap_anglecalc
    theta_cap_anglecalc = np.arctan(dh_anglecalc / r_abs_anglecalc)
    S_G_R_anglecalc = np.array([[np.cos(phi_cap_anglecalc), -np.sin(phi_cap_anglecalc), 0.0],
                      [np.sin(phi_cap_anglecalc), np.cos(phi_cap_anglecalc), 0.0],
                      [0.0, 0.0, 1.0]]).T
    # Transformationsmatrix um z & phi --- [0]=x_R.T, [1]=y_R.T, [2]=z_R.T
    S_G_Rt_anglecalc = np.array([[np.cos(phi_cap_anglecalc) * np.cos(theta_cap_anglecalc), -np.sin(phi_cap_anglecalc), -np.cos(phi_cap_anglecalc) * np.sin(theta_cap_anglecalc)],
                       [np.sin(phi_cap_anglecalc) * np.cos(theta_cap_anglecalc), np.cos(phi_cap_anglecalc), -np.sin(phi_cap_anglecalc) * np.sin(theta_cap_anglecalc)],
                       [np.sin(theta_cap_anglecalc), 0.0, np.cos(theta_cap_anglecalc)]]).T
    # Transformationsmatrix um z & phi, dann y & theta --- [0]=x_Rt.T, [1]=y_Rt.T, [2]=z_Rt.T
    psi_low_anglecalc = get_angle_v_on_plane(z_mauv_anglecalc, np.array(S_G_Rt_anglecalc[2])[np.newaxis].T, np.array(S_G_Rt_anglecalc[1])[np.newaxis].T)
    theta_low_anglecalc = get_angle_v_on_plane(z_mauv_anglecalc, np.array(S_G_R_anglecalc[2])[np.newaxis].T, np.array(S_G_R_anglecalc[0])[np.newaxis].T)
    return phi_cap_anglecalc, theta_cap_anglecalc, psi_low_anglecalc, theta_low_anglecalc


def analyze_measdata_from_file(model_type='log', analyze_tx=[1, 2, 3, 4, 5, 6],  meantype='db_mean', b_onboard=False, measfilename='path'):
    """
    :param analyze_tx:
    :param txpos_tuning:
    :param meantype:
    :return:
    """

    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0

    if b_onboard is True:
        measdata_filename = measfilename
    else:
        measdata_filename = hc_tools.select_file()

    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 0
        measured_wp_list = []

        for i, line in enumerate(measfile):

            if line == '### begin grid settings\n':
                # print('griddata found')
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                # print('Measurement data found')
                continue
            if load_description:
                # print('file description')
                print(line)

            if load_grid_settings and not load_measdata:
                #print(line)

                grid_settings = map(float, line[:-2].split(' '))
                x0 = [grid_settings[0], grid_settings[1], grid_settings[2]]
                xn = [grid_settings[3], grid_settings[4], grid_settings[5]]
                grid_dxdyda = [grid_settings[6], grid_settings[7], grid_settings[8]]
                timemeas = grid_settings[9]

                data_shape_file = []
                for i in range(3):  # range(num_dof)
                    try:
                        shapei = int((xn[i]-x0[i]) / grid_dxdyda[i] + 1)
                    except ZeroDivisionError:
                        shapei = 1
                    data_shape_file.append(shapei)
                # old: data_shape_file = [int((xn[0]-x0[0]) / grid_dxdyda[0] + 1), int((xn[1]-x0[1]) / grid_dxdyda[1] + 1), int((xn[2]-x0[2]) / grid_dxdyda[2] + 1)]
                print('data shape  = ' + str(data_shape_file))

                numtx = int(grid_settings[10])
                txdata = grid_settings[11:(11+4*numtx)]  # urspruenglich [(2+numtx):(2+numtx+3*numtx)]

                # read tx positions
                txpos_list = []
                for itx in range(numtx):
                    itxpos = txdata[3*itx:3*itx+3]  # urspruenglich [2*itx:2*itx+2]
                    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                freqtx_list = []
                for itx in range(numtx):
                    freqtx_list.append(txdata[3*numtx+itx])  # urspruenglich (txdata[2*numtx+itx])
                freqtx = np.asarray(freqtx_list)

                # print out
                print('filename = ' + measdata_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0]*data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdyda = ' + str(grid_dxdyda))
                print('tx_pos = ' + str(txpos_list))
                print('freqtx = ' + str(freqtx))

                startx = x0[0]
                endx = xn[0]
                stepx = data_shape_file[0]

                starty = x0[1]
                endy = xn[1]
                stepy = data_shape_file[1]

                startz = x0[2]
                endz = xn[2]
                stepz = data_shape_file[2]

                xpos = np.linspace(startx, endx, stepx)
                ypos = np.linspace(starty, endy, stepy)
                zpos = np.linspace(startz, endz, stepz)

                # wp_matx, wp_maty, wp_matz = np.meshgrid(xpos, ypos, zpos)  # old wp-creation with z axis being main moving axis
                wp_maty, wp_matz, wp_matx = np.meshgrid(ypos, zpos, xpos)  # put least moving axis second, then second least moving first, then quickest last

                # print(xpos)

            if load_measdata and not load_grid_settings:
                # print('read measdata')

                totnumwp += 1
                meas_data_line = map(float, line[:-2].split(' '))
                meas_data_append_list.append(meas_data_line)

                meas_data_mat_line = np.asarray(meas_data_line)

                measured_wp_list.append(int(meas_data_mat_line[3]))
                num_tx = int(meas_data_mat_line[4])
                num_meas = int(meas_data_mat_line[5])

                first_rss = 6 + num_tx

                meas_data_mat_rss = meas_data_mat_line[first_rss:]

                rss_mat_raw = meas_data_mat_rss.reshape([num_tx, num_meas])  # mat_dim: num_tx x num_meas

                def reject_outliers(data, m=5.):
                    d = np.abs(data - np.median(data))
                    mdev = np.median(d)
                    s = d / mdev if mdev else 0.
                    # print('kicked out samples' + str([s < m]))
                    return data[s < m]

                if meantype is 'lin':
                    rss_mat_lin = 10**(rss_mat_raw/10)
                    mean_lin = np.mean(rss_mat_lin, axis=1)
                    var_lin = np.var(rss_mat_lin, axis=1)
                    mean = 10 * np.log10(mean_lin)
                    var = 10 * np.log10(var_lin)
                else:
                    mean = np.zeros([numtx])
                    var = np.zeros([numtx])
                    for itx in range(numtx):
                        rss_mat_row = reject_outliers(rss_mat_raw[itx, :])
                        # print('kicked out samples:' + str(len(rss_mat_raw[itx, :]) - len(rss_mat_row)))
                        mean[itx] = np.mean(rss_mat_row)
                        var[itx] = np.var(rss_mat_row)
                    # print('var = ' + str(var))
                wp_pos = np.array([meas_data_mat_line[0], meas_data_mat_line[1], meas_data_mat_line[2]])

                # antenna_orientation = np.array([[0.0], [0.64278760968], [0.76604444311]])
                antenna_orientation = np.array([[0.0], [0.0], [1.0]])
                # antenna_orientation = np.array([[0], [0.34202014332], [0.93969262078]])  # todo: Enter antenna orientation for correct parameter calibration here!
                wp_angles = [0.0]*num_tx*4
                for itx in range(num_tx):
                    wp_angles[itx*4:itx*4+4] = get_angles(np.transpose(wp_pos[0:2][np.newaxis]), np.transpose(txpos[itx, 0:2][np.newaxis]), txpos[itx, 2], antenna_orientation, wp_pos[2])
                wp_angles = np.asarray(wp_angles)

                plotdata_line = np.concatenate((wp_pos, mean, var, wp_angles), axis=0)  # -> x,y,a,meantx1,...,meantxn,vartx1,...vartxn
                plotdata_mat_lis.append(plotdata_line)

        measfile.close()

        # data_shape = [data_shape_file[0], data_shape_file[1], data_shape_file[2]]  # data_shape: n_x, n_y, n_z
        data_shape = [data_shape_file[1], data_shape_file[0], data_shape_file[2]]  # data_shape: n_x, n_y, n_z
        plotdata_mat = np.asarray(plotdata_mat_lis)

        """
        Model fit
        """
        if model_type == 'log':
            '''
            def rsm_model(rsm_params, lambda_rsm, gamma_rsm, n_t_rsm, n_r_rsm):
                """Range Sensor Model (RSM) structure."""
                dist_rsm, psi_low_rsm, theta_cap_rsm, theta_low_rsm = rsm_params
                return -20 * np.log10(dist_rsm) + lambda_rsm * dist_rsm + gamma_rsm + np.log10(np.cos(abs(psi_low_rsm))) + n_t_rsm * np.log10(abs(np.cos(theta_cap_rsm))) + n_r_rsm * np.log10(abs(np.cos(theta_cap_rsm + theta_low_rsm)))  # rss in db

            def rsm_model(rsm_params, lambda_rsm, gamma_rsm, n_r_rsm):
                """Range Sensor Model (RSM) structure."""
                dist_rsm, psi_low_rsm, theta_cap_rsm, theta_low_rsm = rsm_params
                return -20 * np.log10(dist_rsm) + lambda_rsm * dist_rsm + gamma_rsm + np.log10(np.cos(abs(psi_low_rsm))) + n_r_rsm * np.log10(abs(np.cos(theta_low_rsm)))  # rss in db
            def rsm_model(rsm_params, lambda_rsm, gamma_rsm, n_t_rsm):
                """Range Sensor Model (RSM) structure."""
                dist_rsm, theta_cap_rsm, psi_low_rsm, theta_low_rsm = rsm_params
                return -20 * np.log10(dist_rsm) + lambda_rsm * dist_rsm + gamma_rsm + 2 * n_t_rsm * np.log10(abs(np.cos(theta_cap_rsm)))  # rss in db
            '''
            def rsm_model(rsm_params, lambda_rsm, gamma_rsm):
                """Range Sensor Model (RSM) structure."""
                dist_rsm, theta_cap_rsm, psi_low_rsm, theta_low_rsm = rsm_params
                return -20 * np.log10(dist_rsm) + lambda_rsm * dist_rsm + gamma_rsm + np.log10(3.83135740649**2) # rss in db
        elif model_type == 'lin':  # todo: OLD: consider linearized Angles when use is desired. decide to not bother linearizing and throw this out of the code.
            def rsm_model(dist_rsm, lambda_rsm, gamma_rsm):
                """Range Sensor Model (RSM) structure."""
                return lambda_rsm * dist_rsm + gamma_rsm  # rss in db

        rdist = []

        calibration_mode = True  # True = measurement had a straight antenna and was performed on transmitter heights

        if calibration_mode:
            lambda_t = []
            gamma_t = []
            n_t = []
            n_r = []
        else:
            lambda_t = []
            gamma_t = []  # todo: enter correct values here for no calibration use!
            n_t = []
            n_r = []

        for itx in analyze_tx:
            rdist_vec = plotdata_mat[:, 0:3] - txpos[itx, 0:3]  # + [0, 0, 0] # r_wp -r_txpos -> dim: num_meas x 2or3 (3 if z is introduced)
            rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  # distance norm: |r_wp -r_txpos| -> dim: num_meas x 1

            if calibration_mode:
                rssdata = plotdata_mat[:, 3+itx]  # rss-mean for each wp
                theta_cap = plotdata_mat[:, 3+num_tx*2+1+itx*4]
                psi_low = plotdata_mat[:, 3+num_tx*2+2+itx*4]
                theta_low = plotdata_mat[:, 3+num_tx*2+3+itx*4]
                rsm_paramtuple = rdist_temp, theta_cap, psi_low, theta_low
                popt, pcov = curve_fit(rsm_model, rsm_paramtuple, rssdata)  #, bounds=([-np.inf, -np.inf, 0], np.inf)
                del pcov

                lambda_t.append(round(popt[0], 7))
                gamma_t.append(round(popt[1], 4))
                #n_t.append(round(popt[2], 4))
                #n_r.append(round(popt[2], 4))

            rdist.append(rdist_temp)

        rdist_temp = np.reshape(rdist, [num_tx, totnumwp])
        if model_type == 'log':
            print('\nVectors for convenient copy/paste')
            print('lambda_t = np.array(' + str(lambda_t) + ')  # ' + measdata_filename)
            print('gamma_t = np.array(' + str(gamma_t) + ')  # ' + measdata_filename)
            print('tx_n = np.array(' + str(n_t) + ')  # ' + measdata_filename)
            print('rx_n = np.array(' + str(n_r) + ')  # ' + measdata_filename)

        elif model_type=='lin':
            print('\nVectors for convenient copy/paste')
            print('lambda_lin = ' + str(lambda_t))
            print('gamma_lin = ' + str(gamma_t))
        
        """
        Plots
        """
        if b_onboard is False:
            x = plotdata_mat[:, 0]
            y = plotdata_mat[:, 1]

            plot_fig0 = True
            if plot_fig0:  # 2D contour plot
                fig = plt.figure(0)
                analyze_tx = [3]
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos)

                    rdata = np.linspace(1, np.max(rdist), num=1000)

                    phi_cap = np.array([0.0] * len(rdata))  # plotdata_mat[:, 3 + num_tx * 2 + 0 + itx * 4]  #
                    theta_cap = np.array([0.0]*len(rdata))  # plotdata_mat[:, 3 + num_tx * 2 + 1 + itx * 4]  #
                    psi_low = np.array([0.0]*len(rdata))  # plotdata_mat[:, 3 + num_tx * 2 + 2 + itx * 4]  #
                    theta_low = np.array([0.0]*len(rdata))  # plotdata_mat[:, 3 + num_tx * 2 + 3 + itx * 4]  #


                    ax.legend(loc='upper right')
                    ax.grid()
                    ax.set_ylim([-110, -10])
                    ax.set_xlabel('Distance [mm]')
                    ax.set_ylabel('RSS [dB]')
                    ax.set_title('RSM for TX# ' + str(itx + 1))

                    # ax.errorbar(rdist[itx], plotdata_mat[:, 3 + itx], yerr=plotdata_mat[:, 3 + num_tx + itx], fmt='ro', markersize='1', ecolor='g', label='Original Data', zorder=1)

                    for iter in np.linspace(0, 600, 7):
                        height_for_plot = iter

                        for i in range(len(rdata)):
                            phi_cap[i], theta_cap[i], psi_low[i], theta_low[i] = get_angles(np.array([[txpos[itx][0]+rdata[i]], [txpos[itx][1]]]), np.array([[txpos[itx][0]], [txpos[itx][1]]]), txpos[itx][2], antenna_orientation, height_for_plot)

                        rsm_paramtuple_plot = rdata, theta_cap, psi_low, theta_low

                        ax.plot(rdata, rsm_model(rsm_paramtuple_plot, lambda_t[itx], gamma_t[itx], n_t[itx]), label='Fitted Curve', zorder=2)  # , n_r[itx]

                    fig.subplots_adjust(hspace=0.4)

                fig = plt.figure(1, figsize=(18, 10))
                # fig = plt.figure(1, figsize=(5,2.5))
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos)

                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]

                    rss_mat_ones = np.ones(np.shape(wp_matx)) * (-200)  # set minimum value for not measured points
                    rss_full_vec = np.reshape(rss_mat_ones, (len(xpos) * len(ypos) * len(zpos), 1))

                    measured_wp_list = np.reshape(measured_wp_list, (len(measured_wp_list), 1))
                    measured_wp_list[:] -= measured_wp_list[0]  # In case that measurements have been selected manually and measurements are not the first ones -> first measurement is meas zero and so on
                    rss_mean = np.reshape(rss_mean, (len(rss_mean), 1))

                    rss_full_vec[measured_wp_list, 0] = rss_mean

                    rss_full_mat = np.reshape(rss_full_vec, data_shape)

                    # mask all points which were not measured
                    # rss_full_mat = np.ma.array(rss_full_mat, mask=rss_full_mat < -199)  # np.ones(np.shape(wp_maty))*(-60)

                    val_sequence = np.linspace(-100, -20, 80 / 5 + 1)

                    # CS = ax.contour(wp_matx[::2, ::2], wp_maty[::2, ::2], rss_full_mat[::2, ::2], val_sequence) # takes every second value
                    CS = ax.contour(wp_matx[0, :, :], wp_maty[0, :, :], rss_full_mat[:, :, 0], val_sequence, cmap=plt.cm.jet, label='RSS Contours')
                    ax.clabel(CS, inline=0, fontsize=10)

                    for itx_plot in analyze_tx:
                        if itx_plot == itx:
                            ax.plot(txpos[itx_plot, 0], txpos[itx_plot, 1], 'or', c='orangered', markersize=15)
                        else:
                            ax.plot(txpos[itx_plot, 0], txpos[itx_plot, 1], 'or')

                    ax.grid()
                    ax.set_xlabel('x [mm]')
                    ax.set_ylabel('y [mm]')
                    ax.axis('equal')
                    ax.set_title('RSS field for TX# ' + str(itx + 1))
                    fig.subplots_adjust(hspace=0.4)

                fig = plt.figure(8)
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos, projection='3d')

                    rss_2_plot = -70
                    rss_2_plot_var = 1
                    same_rss_indexes = np.where(np.logical_and(plotdata_mat[:, 3+itx] <= (rss_2_plot + rss_2_plot_var), plotdata_mat[:, 3+itx] >= (rss_2_plot - rss_2_plot_var)))
                    CS = ax.scatter(plotdata_mat[same_rss_indexes[0], 0], plotdata_mat[same_rss_indexes[0], 1], plotdata_mat[same_rss_indexes[0], 2], label='Scatter 3D for same RSS')  # for coloring kwag: c=plotdata_mat[same_rss_indexes[0], 3+itx]

                    # CS = ax.scatter(wp_matx[:, :, 0], wp_maty[:, :, 0], wp_matz[:, :, 0], val_sequence)
                    ax.clabel(CS, inline=0, fontsize=10)

                    ax.scatter(txpos[itx, 0], txpos[itx, 1], txpos[itx, 2], color='r', s=100)

                    ax.grid()
                    ax.set_xlabel('x [mm]')
                    ax.set_ylabel('y [mm]')
                    ax.axis('equal')
                    ax.set_title('RSS field for TX# ' + str(itx + 1))
                    fig.subplots_adjust(hspace=0.4)

            plot_fig2 = False
            if plot_fig2:
                fig = plt.figure(2)
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos, projection='3d')
                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]
                    ax.plot_trisurf(x, y, rss_mean, cmap=plt.cm.Spectral)
                    ax.grid()
                    ax.set_xlabel('x [mm]')
                    ax.set_ylabel('y [mm]')
                    ax.set_zlabel('rss [dB]')
                    ax.set_zlim([-110, -20])
                    ax.set_title('RSS field for TX# ' + str(itx+1))

            plot_fig3 = False
            if plot_fig3:
                fig = plt.figure(3)
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos, projection='3d')
                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]
                    ax.plot_trisurf(x, y, rss_var, cmap=plt.cm.Spectral)
                    ax.grid()
                    ax.set_xlabel('x [mm]')
                    ax.set_ylabel('y [mm]')
                    ax.set_zlabel('rss_var [dB]')
                    ax.set_title('RSS field variance for TX# ' + str(itx + 1))

            plot_fig4 = False
            if plot_fig4:
                fig = plt.figure(4)
                for itx in analyze_tx:
                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]

                    rdist = np.array(rdist_temp[itx, :], dtype=float)
                    rss_mean = np.array(rss_mean, dtype=float)
                    rss_var = np.array(rss_var, dtype=float)
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111
                    ax = fig.add_subplot(pos)
                    ax.errorbar(rdist, rss_mean, yerr=rss_var,
                                fmt='ro',markersize='1', ecolor='g', label='Original Data')

                    rdata = np.linspace(np.min(rdist), np.max(rdist), num=1000)
                    ax.plot(rdata, rsm_model(rdata, lambda_t[itx], gamma_t[itx]), label='Fitted Curve')
                    ax.legend(loc='upper right')
                    ax.grid()
                    ax.set_ylim([-110, -10])
                    ax.set_xlabel('Distance [mm]')
                    ax.set_ylabel('RSS [dB]')
                    ax.set_title('RSM for TX# ' + str(itx + 1))

            plot_fig5 = False
            if plot_fig5:
                fig = plt.figure(5)
                for itx in analyze_tx:
                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]

                    rdist = np.array(rdist_temp[itx, :], dtype=float)
                    rss_mean = np.array(rss_mean, dtype=float)
                    rss_var = np.array(rss_var, dtype=float)

                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111
                    ax = fig.add_subplot(pos)
                    rssdata = np.linspace(-10, -110, num=1000)
                    ax.plot(rssdata, lambertloc(rssdata, lambda_t[itx], gamma_t[itx]), label='Fitted Curve')
                    ax.plot(rss_mean, rdist, 'r.')
                    ax.grid()
                    ax.set_xlabel('RSS [dB]')
                    ax.set_ylabel('Distance [mm]')

            plot_fig6 = False
            if plot_fig6:
                fig = plt.figure(6)
                for itx in analyze_tx:
                    rss_mean = plotdata_mat[:, 3 + itx]
                    rss_var = plotdata_mat[:, 3 + num_tx + itx]

                    rdist = np.array(rdist_temp[itx, :], dtype=float)
                    rss_mean = np.array(rss_mean, dtype=float)
                    rss_var = np.array(rss_var, dtype=float)

                    r_dist_est = lambertloc(rss_mean, lambda_t[itx], gamma_t[itx])
                    sorted_indices = np.argsort(rdist)
                    r_dist_sort = rdist[sorted_indices]
                    r_dist_est_sort = r_dist_est[sorted_indices]
                    dist_error = r_dist_sort - r_dist_est_sort
                    data_temp = []
                    bin = np.linspace(0, 2000, 21)

                    ibin = 1
                    bin_mean = []
                    bin_var = []
                    for i in range(len(r_dist_sort)):
                        if r_dist_sort[i] >= bin[-1]:
                            break
                        elif bin[ibin-1] <= r_dist_sort[i] < bin[ibin]:
                            data_temp.append(dist_error[i])
                        else:
                            bin_mean_temp = np.mean(data_temp)
                            bin_var_temp = np.var(data_temp)
                            bin_mean.append(bin_mean_temp)
                            bin_var.append(bin_var_temp)
                            # print('bin_high_bound :' + str(bin[ibin]) + ' bin_mean:' + str(bin_mean_temp))
                            data_temp = []  # reset bin
                            data_temp.append(dist_error[i])
                            ibin += 1
                            # print('ibin ' + str(ibin))

                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111
                    ax = fig.add_subplot(pos)
                    # rssdata = np.linspace(-10, -110, num=1000)
                    # ax.plot(rssdata, lambertloc(rssdata, lambda_t[itx], gamma_t[itx]), label='Fitted Curve')

                    # ax.errorbar(bin[1:-1], bin_mean, yerr=bin_var, fmt='ro', ecolor='g', label='Original Data')
                    ax.plot(bin[1:-1], bin_mean, '.')
                    # print('bin_means = ' + str(bin_mean))
                    # print('bin_var = ' + str(bin_var))
                    # ax.plot(r_dist_sort, dist_error, '.')
                    ax.grid()
                    ax.set_xlabel('Distance to tx [mm]')
                    ax.set_ylabel('Error [mm]')

            plot_fig7 = False
            ''' Polar Directional Diagrams for Antenna measurements '''
            if plot_fig7:
                fig = plt.figure(7, figsize=(6, 6))
                for itx in analyze_tx:
                    pos = 321 + itx
                    if len(analyze_tx) == 1:
                        pos = 111

                    ax = fig.add_subplot(pos, projection='polar')

                    rss_max = plotdata_mat[:, 3+itx].max()
                    rss_max_index = np.where(plotdata_mat[:, 3+itx] == rss_max)

                    rss_min = plotdata_mat[:, 3+itx].min()
                    rss_min_index = np.where(plotdata_mat[:, 3 + itx] == rss_min)

                    rss_hpbw = rss_max - 3

                    itx_2 = rss_max_index[0][0]
                    while plotdata_mat[itx_2, 3 + itx] > rss_hpbw:
                        itx_2 += 1
                        if itx_2 == totnumwp:
                            itx_2 = 0
                    else:
                        rss_hpbw_positiveitx_rss = plotdata_mat[itx_2 - 1, 3 + itx]
                        rss_hpbw_positiveitx_rad = plotdata_mat[itx_2 - 1, 2]

                    itx_2 = rss_max_index[0][0]
                    while plotdata_mat[itx_2, 3 + itx] > rss_hpbw:
                        itx_2 -= 1
                        if itx_2 == -1:
                            itx_2 = totnumwp-1
                    else:
                        rss_hpbw_negativeitx_rss = plotdata_mat[itx_2 + 1, 3 + itx]
                        rss_hpbw_negativeitx_rad = plotdata_mat[itx_2 + 1, 2]

                    if abs(rss_hpbw_positiveitx_rss - rss_hpbw_negativeitx_rss) > 0.5:
                        print('~~~~~> Possible ERROR: HPBW-RSS-measurements are far apart: ' + str(abs(rss_hpbw_positiveitx_rss - rss_hpbw_negativeitx_rss)))

                    print('HPBW No1: ' + str(abs(rss_hpbw_positiveitx_rad - rss_hpbw_negativeitx_rad)) + ' rad / ' + str(abs(rss_hpbw_positiveitx_rad - rss_hpbw_negativeitx_rad)*180/np.pi) + ' deg')

                    pot_max_2 = 2*rss_min_index[0][0] - rss_max_index[0][0]
                    if pot_max_2 < 0:
                        pot_max_2 += totnumwp
                    if pot_max_2 >= totnumwp:
                        pot_max_2 -= totnumwp

                    itx_3 = 0
                    if plotdata_mat[pot_max_2, 3+itx] < plotdata_mat[pot_max_2 + 1, 3+itx]:
                        itx_3 += 1
                        while plotdata_mat[pot_max_2 + itx_3, 3+itx] < plotdata_mat[pot_max_2 + itx_3 + 1, 3+itx]:
                            itx_3 += 1
                    else:
                        itx_3 -= 1
                        while plotdata_mat[pot_max_2 + itx_3, 3+itx] < plotdata_mat[pot_max_2 + itx_3 - 1, 3+itx]:
                            itx_3 -= 1

                    rss_max_2 = plotdata_mat[pot_max_2 + itx_3, 3+itx]
                    rss_max_2_index = np.where(plotdata_mat[:, 3 + itx] == rss_max_2)

                    rss_hpbw_2 = rss_max_2 - 3

                    itx_4 = rss_max_2_index[0][0]
                    while plotdata_mat[itx_4, 3 + itx] > rss_hpbw_2:
                        itx_4 += 1
                        if itx_4 == totnumwp:
                            itx_4 = 0
                    else:
                        rss_hpbw_2_positiveitx_rss = plotdata_mat[itx_4 - 1, 3 + itx]
                        rss_hpbw_2_positiveitx_rad = plotdata_mat[itx_4 - 1, 2]

                    itx_4 = rss_max_2_index[0][0]
                    while plotdata_mat[itx_4, 3 + itx] > rss_hpbw_2:
                        itx_4 -= 1
                        if itx_4 == -1:
                            itx_4 = totnumwp - 1
                    else:
                        rss_hpbw_2_negativeitx_rss = plotdata_mat[itx_4 + 1, 3 + itx]
                        rss_hpbw_2_negativeitx_rad = plotdata_mat[itx_4 + 1, 2]

                    if abs(rss_hpbw_2_positiveitx_rss - rss_hpbw_2_negativeitx_rss) > 0.5:
                        print('~~~~~> Possible ERROR: HPBW-RSS-measurements are far apart: ' + str(
                            abs(rss_hpbw_2_positiveitx_rss - rss_hpbw_2_negativeitx_rss)))

                    print('HPBW No2: ' + str(abs(rss_hpbw_2_positiveitx_rad - rss_hpbw_2_negativeitx_rad)) + ' rad / ' + str(abs(rss_hpbw_2_positiveitx_rad - rss_hpbw_2_negativeitx_rad)*180/np.pi) + ' deg')

                    ax.plot(plotdata_mat[:, 2]+np.pi*0.0, plotdata_mat[:, 3+itx], label='Radiation Pattern')
                    ax.set_rmax(rss_max)
                    ax.set_rmin(rss_min)
                    rticks = np.round(np.append(np.linspace(rss_min, rss_max, 5), rss_hpbw), 2)
                    ax.set_rticks(rticks)  # or [rss_max, rss_min]
                    ax.set_rlabel_position(65)
                    # ax.set_rticks([rss_hpbw_negativeitx_rss])  # <- alternative
                    # ax.set_rlabel_position(rss_hpbw_negativeitx_rad*180/np.pi)  # <- alternative

                    # ax.set_title('Radiation Pattern for TX# ' + str(itx + 1), fontsize=20)

        plt.show()

    return lambda_t, gamma_t


"""
Onboard Calibration 
"""


def onboard_cal_param(tx_pos, measdata_filename='meas_data_wburg.txt', param_filename='cal_param.txt'):

    with open(measdata_filename, 'r') as measfile:
        meas_data_append_list = []

        plotdata_mat_lis = []

        totnumwp = 14
        measured_wp_list = []

        numtx = 6

        for i, line in enumerate(measfile):

                #data_shape_file = [int((xn[0]-x0[0]) / grid_dxdy[0] + 1), int((xn[1]-x0[1]) / grid_dxdy[1] + 1)]
                #print('data shape  = ' + str(data_shape_file))

                # read tx positions
                txpos_list = tx_pos
                #for itx in range(numtx):
                #    itxpos = txdata[2*itx:2*itx+2]
                #    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                #freqtx_list = []
                #for itx in range(numtx):
                #    freqtx_list.append(txdata[2*numtx+itx])


                totnumwp += 1

                # way point data - structure 'wp_x, wp_y, num_wp, num_tx, num_meas'
                meas_data_line = map(float, line[0:-3].split(' '))
                meas_data_append_list.append(meas_data_line)
                meas_data_mat_line = np.asarray(meas_data_line)


                measured_wp_list.append(int(meas_data_mat_line[2]))
                num_tx = int(meas_data_mat_line[3])
                num_meas = int(meas_data_mat_line[4])

                # line is follows by freq_tx of all transceiver
                first_rss = 5 + num_tx

                # rss data - str_rss structure: 'ftx1.1, ftx1.2, [..] ,ftx1.n, ftx2.1, ftx2.2, [..], ftx2.n
                meas_data_mat_rss = meas_data_mat_line[first_rss:]

                rss_mat_raw = meas_data_mat_rss.reshape([num_tx, num_meas])

                def reject_outliers(data, m=5.):
                    d = np.abs(data - np.median(data))
                    mdev = np.median(d)
                    s = d / mdev if mdev else 0.
                    # print('kicked out samples' + str([s < m]))
                    return data[s < m]

                mean = np.zeros([numtx])
                var = np.zeros([numtx])
                for itx in range(numtx):
                    rss_mat_row = rss_mat_raw[itx, :]  # reject_outliers(rss_mat_raw[itx, :])

                    # print('kicked out samples:' + str(len(rss_mat_raw[itx, :]) - len(rss_mat_row)))
                    mean[itx] = np.mean(rss_mat_row)
                    var[itx] = np.var(rss_mat_row)
                # print('var = ' + str(var))
                wp_pos = [meas_data_mat_line[0], meas_data_mat_line[1]]

                plotdata_line = np.concatenate((wp_pos, mean, var), axis=0)
                plotdata_mat_lis.append(plotdata_line)

        measfile.close()

        # data_shape = [data_shape_file[1], data_shape_file[0]]
        plotdata_mat = np.asarray(plotdata_mat_lis)
        #print('Plot data mat =' + str(plotdata_mat))

        """
        Model fit
        """

        def rsm_model(dist_rsm, alpha_rsm, gamma_rsm):
            """Range Sensor Model (RSM) structure."""
            return -20 * np.log10(dist_rsm) - alpha_rsm * dist_rsm - gamma_rsm  # rss in db

        alpha = []
        gamma = []
        rdist = []

        for itx in range(numtx):
            rdist_vec = plotdata_mat[:, 0:2] - txpos[itx, 0:2]  # + [250 , 30] # r_wp -r_txpos

            rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  # distance norm: |r_wp -r_txpos|

            rssdata = plotdata_mat[:, 2+itx]  # rss-mean for each wp
            popt, pcov = curve_fit(rsm_model, rdist_temp, rssdata)
            del pcov

            alpha.append(round(popt[0], 4))
            gamma.append(round(popt[1], 4))
            # print('tx #' + str(itx+1) + ' alpha= ' + str(alpha[itx]) + ' gamma= ' + str(gamma[itx]))
            rdist.append(rdist_temp)
            #print('itx = ' + str(itx))
        #print('rdist = ' + str(rdist))

        with open(param_filename, 'w') as paramfile:
            for itx in range(numtx):
                paramfile.write(str(alpha[itx])+' ')
            paramfile.write('\n')

            for itx in range(numtx):
                paramfile.write(str(gamma[itx])+' ')
            paramfile.write('\n')
            paramfile.close()

        print('\nVectors for convenient copy/paste')
        print('alpha = ' + str(alpha))
        print('gamma = ' + str(gamma))

    return alpha, gamma


def get_cal_param_from_file(param_filename='cal_param.txt'):
    with open(param_filename, 'r') as param_file:
        param_list = []
        for i, line in enumerate(param_file):

            param_line = map(float, line[:-2].split(' '))

            param_list.append(param_line)

    alpha = param_list[0]
    gamma = param_list[1]

    return alpha, gamma


def lambertloc(rss, alpha, gamma):
    """Inverse function of the RSM. Returns estimated range in [cm].

    Keyword arguments:
    :param rss -- received power values [dB]
    :param alpha
    :param gamma
    """
    z = 20 / (np.log(10) * alpha) * lambertw(np.log(10) * alpha / 20 * np.exp(-np.log(10) / 20 * (rss + gamma)))
    return z.real  # [mm]

