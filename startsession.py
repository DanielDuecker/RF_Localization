import rf
import numpy as np
cal = rf.CalEar(433.9e6)
# cal.plot_psd()

#cal.plot_multi_rss_live(433.91e6, 434.16e6, 2e4)
freqtx = [433.7e6]  # [433.9e6] ##  [434.16e6] #
#cal.get_max_rss_in_freqspan(freqtx, 2e4)
# cal.make_test(5.0)
#cal.plot_multi_rss_live(433.91e6, 434.16e6)


#rss, var = cal.measure_rss_var(freqtx, 2e4, 10.0)
#alpha, xi = cal.get_model(rss, var)

#print ('Tx freq : ' + str(freqtx/1e6) + 'MHz, alpha = ' + str(alpha) + ' xi = ' + str(xi))


# messung kw 43    alpha = 0.123262770536  , xi = 28.4340027272 @ 433,91MHz
# messung 25.10.16 alpha = 0.0939750617382 , xi = 28.2926626424 @ 433,91MHz
# messung 25.10.16 alpha = 0.103343746736  , xi = 27.3188771771 @ 433,91MHz (rot)
#
# messung 25.10.16 alpha = 0.0874479464684 , xi = 23.8654845692 @ 434,16MHz
# messung 25.10.16 alpha = 0.0880381844913 , xi = 23.8442108093 @ 434,16MHz

# messung laengs
# messung 26.10.16 alpha = 0.12615852725  , xi = 28.2177151396 @ 433,91MHz
# messung 26.10.16 alpha = 0.117592701848 , xi = 11.9628114874 @ 434,16MHz


# messung 28.10.16 alpha = ... , xi = ... @ 433,91MHz

# tx @ 433,91Mhz (rot)
# messung 28.10.16 alpha = 0.110816801657 , xi = 26.8577561834 @ 433,91MHz (1,10cm)
# messung 28.10.16 alpha = 0.0969777958544 , xi = 28.2082988357 @ 433,91MHz (2,10cm)
# messung 28.10.16 alpha = 0.0858214838313 , xi = xi = 29.225175449 @ 433,91MHz (3,10cm)
# messung 28.10.16 alpha = 0.143116087889 , xi = xi = 27.8787996741 @ 433,91MHz (4,10cm)
# messung 28.10.16 alpha = 0.113874101685 , xi = xi = 27.8255087801 @ 433,91MHz (5,5cm)
# messung 28.10.16 alpha = 0.108715953824 , xi = xi = 27.9195492557 @ 433,91MHz (6,5cm)

# tx @ 434,1Mhz

# messung 28.10.16 alpha = 0.132392992151, xi = 11.8097453463 @ 434,1MHz (7,10cm)
# messung 28.10.16 alpha = 0.136905430224, xi = 11.4113631184 @ 434,1MHz (8,10cm)
# messung 28.10.16 alpha = 0.120072508061, xi = 11.2530408276 @ 434,1MHz (9,10cm)
# messung 28.10.16 alpha = 0.125679169301, xi = 11.0395325748 @ 434,1MHz (10,10cm)
# messung 28.10.16 alpha = 0.136580151677, xi = 10.4178109223 @ 434,1MHz (11,5cm)
# messung 28.10.16 alpha = 0.133571421412, xi = 10.649067175 @ 434,1MHz (12,5cm)

# tx @ 433,7Mhz (RF-Explorer)

# messung 28.10.16 alpha = 0.133571421412, xi = 10.649067175 @ 433,7MHz (13,10cm)
# messung 28.10.16 alpha = 0.101507983364, xi = 45.2926705735 @ 433,7MHz (14,10cm)
# messung 28.10.16 alpha = 0.12121142884, xi = 42.8913406727 @ 433,7MHz (15,10cm)
# messung 28.10.16 alpha = 0.0941960826825, xi = 47.6805702706 @ 433,7MHz (16,10cm)
# messung 28.10.16 alpha = 0.114243243761, xi = 47.7769228175 @ 433,7MHz (17,5cm)
# messung 28.10.16 alpha = 0.114296800679, xi = 47.6663384091 @ 433,7MHz (18,5cm)
# messung 28.10.16 alpha = 0.0878191181987, xi = 51.5224892245 @ 433,7MHz (19,10cm)
# messung 28.10.16 alpha = 0.0782452466389, xi = 51.8319571051 @ 433,7MHz (20,10cm)
# messung 28.10.16 alpha = 0.126511376541, xi = 50.6876178018 @ 433,7MHz (21,5cm)
# messung 28.10.16 alpha = 0.137397311502, xi = 50.6030796818 @ 433,7MHz (22,5cm)


alpha = [0.12615852725, 0.117592701848, 0.114243243761]
# xi = [28.2177151396, 11.9628114874, 47.7769228175]
# xi = [44.7672072423, 11.9628114874, 47.7769228175]  # after calibration of tx0
# xi = [11.7672072423, 11.9628114874, 47.7769228175]  # tuning tx0
xi = [10.6176901836, 12.9628114874, 23.2984322235]  # tuning tx0
freqtx = [433.91e6, 434.16e6, 433.7e6]
freqspan = 2e4
freqcenter = 434.0e6
txpos = np.array([[0.0, 0.0],
                  [80.0, 0.0],
                  [40.0, 62.0]])  #

# mat_test = np.array([[1,2], [3, 4]])
# print('mat_out' + str(mat_test))

# print('txpos' + str(txpos))
# freqmax, rssmax = cal.get_max_rss_in_freqspan(freqtx, 2e4)
# print ('freq_max ' + str(freqmax) + ' rss_max ' + str(rssmax))

loc = rf.LocEar(alpha, xi, freqtx, freqspan, freqcenter)
# loc.plot_multi_dist_live(freqtx)
# loc.calibrate(2)
x0 = np.array([0, 40])
loc.map_path_ekf(x0, txpos)
