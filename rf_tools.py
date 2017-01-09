import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import curve_fit
#from scipy.special import lambertw


"""
independent methods related to the gantry
"""


def wp_generator(wp_filename='wplist.txt', x0=[0, 0], xn=[1200, 1200], steps=[7, 7], timemeas=10.0, show_plot=False):
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
    if show_plot:
        plt.figure()
        plt.plot(wp_mat[:, 0], wp_mat[:, 1], '.-')
        plt.show()

    return wp_filename  # file output [line#, x, y, time]


def analyse_measdata_from_file(measdata_filename, txpos, txpos_offset=[0, 0], freqtx=[433.9e6, 434.1e6],meantype='db_mean'):
    # write header to measurement file
    with open(measdata_filename, 'r') as measfile:
        plotdata_mat_lis = []

        for i, line in enumerate(measfile):
            if i >= 3:  # ignore header (first 3 lines)

                meas_data_list = map(float, line[0:-3].split(', '))
                #print(meas_data_list)

                meas_data_mat_line = np.asarray(meas_data_list)
                #print(meas_data_mat_line)

                # print ('x = ' + str(meas_data_mat_line[0]) + ' y= ' + str(meas_data_mat_line[1]))

                #wp_meas_lis.append([meas_data_mat_line[0], meas_data_mat_line[1], meas_data_mat_line[2]])
                #print ('wp_lis ' + str(wp_meas_lis))
                #print ('wp_lis_shape ' + str(wp_meas_lis.shape))
                num_wp = int(meas_data_mat_line[2])
                num_tx = int(meas_data_mat_line[3])
                num_meas = int(meas_data_mat_line[4])
                freq_vec = []
                # @todo add numtx to data file
                first_rss = 5 + num_tx

                meas_data_mat_rss = meas_data_mat_line[first_rss:]
                #meas_data_mat_rss = meas_data_mat_line[first_rss:-1]  # select only rss data

                #print('num_tx ' + str(num_tx))
                #print('num_meas ' + str(num_meas))

                #print('rss_mat.shape: ' + str(meas_data_mat_rss.shape))
                rss_mat = meas_data_mat_rss.reshape([num_tx, num_meas])

                # print(meas_data_mat_line)

                #print (rss_mat)
                #print (rss_mat.shape)
                if meantype is 'lin':
                    rss_mat_lin = 10**(rss_mat/10)
                    mean_lin = np.mean(rss_mat_lin, axis=1)
                    var_lin = np.var(rss_mat_lin, axis=1)
                    mean = 10 * np.log10(mean_lin)
                    var = 10 * np.log10(var_lin)
                else:
                    mean = np.mean(rss_mat, axis=1)
                    var = np.var(rss_mat, axis=1)
                #print ('mean: ' + str(mean))
                #print ('var: ' + str(var))
                wp = [meas_data_mat_line[0], meas_data_mat_line[1]]

                plotdata_line = np.concatenate((wp, mean, var), axis=1)
                #print (plotdata_line)
                plotdata_mat_lis.append(plotdata_line)
                #plotdata_mat = np.append(plotdata_mat, plotdata_line,axis=1)

        measfile.close()
        totnumwp = num_wp + 1  # counting starts with zero

        plotdata_mat = np.asarray(plotdata_mat_lis)
        print('Number of gridpoints: ' + str(plotdata_mat.shape[0]))
        # print (plotdata_mat)


        """
        Model fit
        """

        def rsm_model(dist, alpha, xi):
            """Range Sensor Model (RSM) structure."""
            return -20 * np.log10(dist) - alpha * dist - xi  # rss in db

        txpos = txpos + txpos_offset  # necessary since gantry frame and the tx-frame are shifted

        alpha = []
        xi = []
        rdist = []#np.ones([totnumwp, num_tx])
        #print (rdist.shape)

        for itx in range(num_tx):
            rdist_vec = plotdata_mat[:, 0:2] - txpos[itx, 0:2]  # r_wp -r_txpos

            rdist_temp = np.asarray(np.linalg.norm(rdist_vec, axis=1))  #  distance norm: |r_wp -r_txpos|

            rssdata = plotdata_mat[:, 2+itx]  # rss-mean for each wp

            #print('itx ' + str(itx) + ' rdist ' + str(rdist_temp))
            #plt.figure(itx)
            #plt.plot(rdist_temp, rssdata,'.')
            #plt.xlabel('dist')
            #plt.ylabel('rss')
            #print('itx ' + str(itx) + ' rss ' + str(rssdata))
            popt, pcov = curve_fit(rsm_model, rdist_temp, rssdata)
            #print('itx = ' + str(itx) + ' popt = ' + str(popt))
            del pcov
            alpha.append(popt[0])
            xi.append(popt[1])
            print('tx #' + str(itx+1) + ' alpha= ' + str(alpha[itx]) + ' xi= ' + str(xi[itx]))
            rdist.append(rdist_temp)

        rdist_temp = np.reshape(rdist,[num_tx, totnumwp])

        fig = plt.figure(1)
        for itx in range(num_tx):
            rss_mean = plotdata_mat[:, 2+itx]
            rss_var = plotdata_mat[:, 2+num_tx+itx]

            rdist = np.array(rdist_temp[itx,:], dtype=float)
            rss_mean = np.array(rss_mean, dtype=float)
            rss_var = np.array(rss_var, dtype=float)
            pos = 221 + itx
            ax = fig.add_subplot(pos)
            ax.errorbar(rdist, rss_mean, yerr=rss_var,
                         fmt='ro', ecolor='g', label='Original Data')

            #print ('alpha = %s , xi = %s' % (alpha, xi))

            rdata = np.linspace(np.min(rdist), np.max(rdist), num=1000)
            ax.plot(rdata, rsm_model(rdata, alpha[itx], xi[itx]), label='Fitted Curve')
            ax.legend(loc='upper right')
            ax.grid()
            ax.set_xlabel('Distance [mm]')
            ax.set_ylabel('RSS [dB]')
            ax.set_title('RSM for TX# ' + str(itx+1))




        """
        Plots
        """
        x = plotdata_mat[:, 0]
        y = plotdata_mat[:, 1]

        fig = plt.figure(2)

        for itx in range(num_tx):
            pos =221 + itx

            ax = fig.add_subplot(pos, projection='3d')
            ax.plot_trisurf(x, y, plotdata_mat[:, 2 + itx], cmap=plt.cm.Spectral)
            ax.grid()
            ax.set_xlabel('x [mm]')
            ax.set_ylabel('y [mm]')
            ax.set_zlabel('rss [dB]')
            ax.set_title('RSS field for TX# ' + str(itx+1))
        plt.show()
