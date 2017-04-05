import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
from scipy.special import lambertw

import hippocampus_toolbox as hc_tools
"""
independent methods related to the gantry
"""


def wp_generator(wp_filename, x0=[0, 0], xn=[1200, 1200], grid_dxdy=[50, 50], timemeas=10.0, show_plot=False):
    """
    :param wp_filename:
    :param x0: [x0,y0] - start position of the grid
    :param xn: [xn,yn] - end position of the grid
    :param steps: [numX, numY] - step size
    :param timemeas: - time [s] to wait at each position for measurements
    :return: wp_mat [x, y, t]
    """
    steps = [(xn[0]-x0[0])/grid_dxdy[0]+1, (xn[1]-x0[1])/grid_dxdy[1]+1]
    print('wp-grid_shape = ' + str(steps))

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

    # wp_filename = hc_tools.save_as_dialog('Save way point list as...')
    with open(wp_filename, 'w') as wpfile:
        wpfile.write('Way point list \n')
        wpfile.write('### begin grid settings\n')
        wpfile.write(str(x0[0]) + ', ' + str(x0[1]) + ', ' +
                     str(xn[0]) + ', ' + str(xn[1]) + ', ' +
                     str(grid_dxdy[0]) + ', ' + str(grid_dxdy[1]) + ', ' +
                     str(timemeas) +
                     '\n')

        wpfile.write('### begin wp_list\n')
        for i in range(wp_mat.shape[0]):
            wpfile.write(str(i) + ', ' + str(wp_mat[i, 0]) + ', ' + str(wp_mat[i, 1]) + ', ' + str(wp_mat[i, 2]) + '\n')
        wpfile.close()
    if show_plot:
        plt.figure()
        plt.plot(wp_mat[:, 0], wp_mat[:, 1], '.-')
        plt.show()
    print('Way point generator terminated!')
    return wp_filename  # file output [line#, x, y, time]


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
                grid_settings = map(float, line.split(','))
                x0 = [grid_settings[0], grid_settings[1]]
                xn = [grid_settings[2], grid_settings[3]]
                grid_dxdy = [grid_settings[4], grid_settings[5]]
                timemeas = grid_settings[6]

                data_shape = [xn[0]/grid_dxdy[0]+1, xn[1]/grid_dxdy[1]+1]
                print('wp-grid_shape = ' + str(data_shape))

            if load_wplist and not load_grid_settings:
                #print('read wplist')
                wp_append_list.append(map(float, line.split(',')))

        print(str(np.asarray(wp_append_list)))
        wp_data_mat = np.asarray(wp_append_list)

        wpfile.close()
        return wp_data_mat, x0, xn, grid_dxdy, timemeas, data_shape


def write_measfile_header(ofile, file_description, x0, xn, grid_dxdy, timemeas, numtx, tx_abs_pos, freqtx):
    txdata = str(numtx) + ', '
    for itx in range(numtx):
        txpos = tx_abs_pos[itx]
        txdata += str(txpos[0]) + ', ' + str(txpos[1]) + ', '
    for itx in range(numtx):
        txdata += str(freqtx[itx]) + ', '

    print('txdata = ' + txdata)

    ofile.write('Way point list \n')
    ofile.write(file_description)
    ofile.write('### begin grid settings\n')
    ofile.write(str(x0[0]) + ', ' + str(x0[1]) + ', ' +
                 str(xn[0]) + ', ' + str(xn[1]) + ', ' +
                 str(grid_dxdy[0]) + ', ' + str(grid_dxdy[1]) + ', ' +
                 str(timemeas) + ', ' + txdata +
                 '\n')
    return True


def analyze_measdata_from_file(analyze_tx, txpos_tuning, meantype='db_mean'):
    """

    :param analyze_tx:
    :param txpos_tuning:
    :param meantype:
    :return:
    """

    analyze_tx[:] = [x - 1 for x in analyze_tx]  # substract -1 as arrays begin with index 0

    measdata_filename = hc_tools.select_file()


    with open(measdata_filename, 'r') as measfile:
        load_description = True
        load_grid_settings = False
        load_measdata = False
        meas_data_append_list = []

        plotdata_mat_lis = []

        for i, line in enumerate(measfile):

            if line == '### begin grid settings\n':
                #print('griddata found')
                load_description = False
                load_grid_settings = True
                load_measdata = False
                continue
            elif line == '### begin measurement data\n':
                load_description = False
                load_grid_settings = False
                load_measdata = True
                #print('Measurement data found')
                continue
            if load_description:
                #print('file description')
                print(line)

            if load_grid_settings and not load_measdata:
                #print(line)

                grid_settings = map(float, line[0:-3].split(','))
                x0 = [grid_settings[0], grid_settings[1]]
                xn = [grid_settings[2], grid_settings[3]]
                grid_dxdy = [grid_settings[4], grid_settings[5]]
                timemeas = grid_settings[6]

                data_shape_file = [int((xn[0]-x0[0]) / grid_dxdy[0] + 1), int((xn[1]-x0[1]) / grid_dxdy[1] + 1)]

                numtx = int(grid_settings[7])
                txdata = grid_settings[8:8+3*numtx]

                # read tx positions
                txpos_list = []
                for itx in range(numtx):
                    itxpos = txdata[2*itx:2*itx+2]
                    txpos_list.append(itxpos)
                txpos = np.asarray(txpos_list)

                # read tx frequencies
                freqtx_list = []
                for itx in range(numtx):
                    freqtx_list.append(txdata[2*numtx+itx])
                freqtx = np.asarray(freqtx_list)

                # print out
                print('filename = ' + measdata_filename)
                print('num_of_gridpoints = ' + str(data_shape_file[0]*data_shape_file[1]))
                print('x0 = ' + str(x0))
                print('xn = ' + str(xn))
                print('grid_shape = ' + str(data_shape_file))
                print('steps_dxdy = ' + str(grid_dxdy))
                print('tx_pos = ' + str(txpos_list))
                print('freqtx = ' + str(freqtx))

            if load_measdata and not load_grid_settings:
                #print('read measdata')
                meas_data_line = map(float, line[0:-3].split(', '))
                meas_data_append_list.append(meas_data_line)


                meas_data_mat_line = np.asarray(meas_data_line)

                num_wp = int(meas_data_mat_line[2])
                num_tx = int(meas_data_mat_line[3])
                num_meas = int(meas_data_mat_line[4])

                # @todo add numtx to data file
                freq_vec = freqtx

                first_rss = 5 + num_tx

                meas_data_mat_rss = meas_data_mat_line[first_rss:]

                rss_mat = meas_data_mat_rss.reshape([num_tx, num_meas])

                if meantype is 'lin':
                    rss_mat_lin = 10**(rss_mat/10)
                    mean_lin = np.mean(rss_mat_lin, axis=1)
                    var_lin = np.var(rss_mat_lin, axis=1)
                    mean = 10 * np.log10(mean_lin)
                    var = 10 * np.log10(var_lin)
                else:
                    mean = np.mean(rss_mat, axis=1)
                    var = np.var(rss_mat, axis=1)
                    # print('var = ' + str(var))
                wp_pos = [meas_data_mat_line[0], meas_data_mat_line[1]]

                plotdata_line = np.concatenate((wp_pos, mean, var), axis=1)

                plotdata_mat_lis.append(plotdata_line)

        measfile.close()

        totnumwp = num_wp + 1  # counting starts with zero

        plotdata_mat = np.asarray(plotdata_mat_lis)
        #print('Number of gridpoints: ' + str(plotdata_mat.shape[0]))



        """
        Model fit
        """

        def rsm_model(dist, alpha, gamma):
            """Range Sensor Model (RSM) structure."""
            return -20 * np.log10(dist) - alpha * dist - gamma  # rss in db

        txpos = txpos + txpos_tuning  # necessary since gantry frame and the tx-frame are shifted

        alpha = []
        gamma = []
        rdist = []

        for itx in analyze_tx:
            rdist_vec = plotdata_mat[:, 0:2] - txpos[itx, 0:2]  # r_wp -r_txpos

            rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  # distance norm: |r_wp -r_txpos|

            rssdata = plotdata_mat[:, 2+itx]  # rss-mean for each wp

            popt, pcov = curve_fit(rsm_model, rdist_temp, rssdata)
            del pcov

            alpha.append(popt[0])
            gamma.append(popt[1])
            # print('tx #' + str(itx+1) + ' alpha= ' + str(alpha[itx]) + ' gamma= ' + str(gamma[itx]))
            rdist.append(rdist_temp)

        rdist_temp = np.reshape(rdist, [num_tx, totnumwp])

        print('\nVectors for convenient copy/paste')
        print('alpha = ' + str(alpha))
        print('gamma = ' + str(gamma))

        """
        Plots
        """
        x = plotdata_mat[:, 0]
        y = plotdata_mat[:, 1]



        fig = plt.figure(6)
        for itx in analyze_tx:
            pos = 221 + itx
            if len(analyze_tx) == 1:
                pos = 111



            ax = fig.add_subplot(pos)
            rss_mean = plotdata_mat[:, 2 + itx]
            rss_var = plotdata_mat[:, 2 + num_tx + itx]

            #data_shape = [16,30]
            #data_shape = [31, 59]#
            #print('shape : ' + str(np.shape(x)))


            #print('shape_from_file ' + str(data_shape_file))
            data_shape = [data_shape_file[1], data_shape_file[0]]

            xx = np.reshape(x, data_shape)
            yy = np.reshape(y, data_shape)
            rss = np.reshape(rss_mean, data_shape)

            val_sequence = np.linspace(-100,-20, 80/5+1)
            CS = ax.contour(xx, yy, rss, val_sequence)
            ax.clabel(CS, inline=0, fontsize=10)
            for itx_plot in analyze_tx:
                ax.plot(txpos[itx_plot-1, 0], txpos[itx_plot-1, 1], 'or')

            ax.grid()
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            # ax.axis('equal')

            ax.set_title('RSS field for TX# ' + str(itx + 1))

        plot_fig1 = True
        if plot_fig1:
            fig = plt.figure(1)
            for itx in analyze_tx:
                pos = 221 + itx
                if len(analyze_tx) == 1:
                    pos = 111

                ax = fig.add_subplot(pos, projection='3d')
                rss_mean = plotdata_mat[:, 2 + itx]
                rss_var = plotdata_mat[:, 2 + num_tx + itx]
                ax.plot_trisurf(x, y, rss_mean, cmap=plt.cm.Spectral)
                ax.grid()
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_zlabel('rss [dB]')
                ax.set_zlim([-110, -20])
                ax.set_title('RSS field for TX# ' + str(itx+1))

        plot_fig2 = True
        if plot_fig2:
            fig = plt.figure(2)
            for itx in analyze_tx:
                pos = 221 + itx
                if len(analyze_tx) == 1:
                    pos = 111

                ax = fig.add_subplot(pos, projection='3d')
                rss_mean = plotdata_mat[:, 2 + itx]
                rss_var = plotdata_mat[:, 2 + num_tx + itx]
                ax.plot_trisurf(x, y, rss_var, cmap=plt.cm.Spectral)
                ax.grid()
                ax.set_xlabel('x [mm]')
                ax.set_ylabel('y [mm]')
                ax.set_zlabel('rss_var [dB]')
                ax.set_title('RSS field variance for TX# ' + str(itx + 1))

        plot_fig3 = True
        if plot_fig3:
            fig = plt.figure(3)
            for itx in analyze_tx:
                rss_mean = plotdata_mat[:, 2 + itx]
                rss_var = plotdata_mat[:, 2 + num_tx + itx]

                rdist = np.array(rdist_temp[itx, :], dtype=float)
                rss_mean = np.array(rss_mean, dtype=float)
                rss_var = np.array(rss_var, dtype=float)
                pos = 221 + itx
                if len(analyze_tx) == 1:
                    pos = 111
                ax = fig.add_subplot(pos)
                ax.errorbar(rdist, rss_mean, yerr=rss_var,
                            fmt='ro', ecolor='g', label='Original Data')

                rdata = np.linspace(np.min(rdist), np.max(rdist), num=1000)
                ax.plot(rdata, rsm_model(rdata, alpha[itx], gamma[itx]), label='Fitted Curve')
                ax.legend(loc='upper right')
                ax.grid()
                ax.set_ylim([-110, -10])
                ax.set_xlabel('Distance [mm]')
                ax.set_ylabel('RSS [dB]')
                ax.set_title('RSM for TX# ' + str(itx + 1))

        plot_fig4 = False
        if plot_fig4:
            fig = plt.figure(4)
            for itx in analyze_tx:
                rss_mean = plotdata_mat[:, 2 + itx]
                rss_var = plotdata_mat[:, 2 + num_tx + itx]

                rdist = np.array(rdist_temp[itx, :], dtype=float)
                rss_mean = np.array(rss_mean, dtype=float)
                rss_var = np.array(rss_var, dtype=float)

                pos = 221 + itx
                if len(analyze_tx) == 1:
                    pos = 111
                ax = fig.add_subplot(pos)
                rssdata = np.linspace(-10, -110, num=1000)
                ax.plot(rssdata, lambertloc(rssdata, alpha[itx], gamma[itx]), label='Fitted Curve')
                ax.plot(rss_mean, rdist, 'r.')
                ax.grid()
                ax.set_xlabel('RSS [dB]')
                ax.set_ylabel('Distance [mm]')


        plot_fig5 = False
        if plot_fig5:
            fig = plt.figure(5)
            for itx in analyze_tx:
                rss_mean = plotdata_mat[:, 2 + itx]
                rss_var = plotdata_mat[:, 2 + num_tx + itx]

                rdist = np.array(rdist_temp[itx, :], dtype=float)
                rss_mean = np.array(rss_mean, dtype=float)
                rss_var = np.array(rss_var, dtype=float)

                r_dist_est = lambertloc(rss_mean, alpha[itx], gamma[itx])
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
                        #print('bin_high_bound :' + str(bin[ibin]) + ' bin_mean:' + str(bin_mean_temp))
                        data_temp = []  # reset bin
                        data_temp.append(dist_error[i])
                        ibin += 1
                        #print('ibin ' + str(ibin))

                pos = 221 + itx
                if len(analyze_tx) == 1:
                    pos = 111
                ax = fig.add_subplot(pos)
                #rssdata = np.linspace(-10, -110, num=1000)
                #ax.plot(rssdata, lambertloc(rssdata, alpha[itx], gamma[itx]), label='Fitted Curve')


                #ax.errorbar(bin[1:-1], bin_mean, yerr=bin_var, fmt='ro', ecolor='g', label='Original Data')
                ax.plot(bin[1:-1], bin_mean, '.')
                #print('bin_means = ' + str(bin_mean))
                #print('bin_var = ' + str(bin_var))
                #ax.plot(r_dist_sort, dist_error, '.')
                ax.grid()
                ax.set_xlabel('Distance to tx [mm]')
                ax.set_ylabel('Error [mm]')

    plt.show()

    return alpha, gamma


def lambertloc(rss, alpha, gamma):
    """Inverse function of the RSM. Returns estimated range in [cm].

    Keyword arguments:
    :param rss -- received power values [dB]
    :param alpha
    :param xi
    """
    z = 20 / (np.log(10) * alpha) * lambertw(np.log(10) * alpha / 20 * np.exp(-np.log(10) / 20 * (rss + gamma)))
    return z.real  # [mm]


