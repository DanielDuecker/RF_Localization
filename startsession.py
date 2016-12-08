import rf
import numpy as np

freqtx = [433.7e6, 433.9e6, 434.16e6]






cal = rf.CalEar(freqtx)

#cal.print_pxx_density()

cal.plot_psd()

#cal.wp_generator('test_wp_list.txt')
#cal.measure_at_waypoint('test_wp_list.txt', 'meas_test.txt')

#cal.get_performance()


#cal.plot_txrss_live()

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
# messung 28.10.16 alpha = 0.110816801657 , xi = 26.857756183 @ 433,91MHz (1,10cm)
# messung 28.10.16 alpha = 0.096977795854 , xi = 28.208298835 @ 433,91MHz (2,10cm)
# messung 28.10.16 alpha = 0.085821483831 , xi = 29.225175449 @ 433,91MHz (3,10cm)
# messung 28.10.16 alpha = 0.143116087889 , xi = 27.878799674 @ 433,91MHz (4,10cm)
# messung 28.10.16 alpha = 0.113874101685 , xi = 27.825508780 @ 433,91MHz (5,5cm)
# messung 28.10.16 alpha = 0.108715953824 , xi = 27.919549255 @ 433,91MHz (6,5cm)

# tx @ 434,1Mhz

# messung 28.10.16 alpha = 0.132392992151, xi = 11.809745346 @ 434,1MHz (7,10cm)
# messung 28.10.16 alpha = 0.136905430224, xi = 11.411363118 @ 434,1MHz (8,10cm)
# messung 28.10.16 alpha = 0.120072508061, xi = 11.253040827 @ 434,1MHz (9,10cm)
# messung 28.10.16 alpha = 0.125679169301, xi = 11.039532574 @ 434,1MHz (10,10cm)
# messung 28.10.16 alpha = 0.136580151677, xi = 10.417810922 @ 434,1MHz (11,5cm)
# messung 28.10.16 alpha = 0.133571421412, xi = 10.649067175 @ 434,1MHz (12,5cm)

# tx @ 433,7Mhz (RF-Explorer)

# messung 28.10.16 alpha = 0.13357142141, xi = 10.649067175 @ 433,7MHz (13,10cm)
# messung 28.10.16 alpha = 0.10150798336, xi = 45.292670573 @ 433,7MHz (14,10cm)
# messung 28.10.16 alpha = 0.12121142884, xi = 42.891340672 @ 433,7MHz (15,10cm)
# messung 28.10.16 alpha = 0.09419608268, xi = 47.680570270 @ 433,7MHz (16,10cm)
# messung 28.10.16 alpha = 0.11424324376, xi = 47.776922817 @ 433,7MHz (17,5cm)
# messung 28.10.16 alpha = 0.11429680067, xi = 47.666338409 @ 433,7MHz (18,5cm)
# messung 28.10.16 alpha = 0.08781911819, xi = 51.522489224 @ 433,7MHz (19,10cm)
# messung 28.10.16 alpha = 0.07824524663, xi = 51.831957105 @ 433,7MHz (20,10cm)
# messung 28.10.16 alpha = 0.12651137654, xi = 50.687617801 @ 433,7MHz (21,5cm)
# messung 28.10.16 alpha = 0.13739731150, xi = 50.603079681 @ 433,7MHz (22,5cm)

# manually tuned parameter
alpha = [0.12615852725, 0.117592701848, 0.114243243761]
xi = [10.6176901836, 12.9628114874, 23.2984322235]
freqtx = [433.91e6, 434.16e6, 433.7e6]
freqspan = 2e4
freqcenter = 434.0e6


#loc = rf.LocEar(alpha, xi, freqtx, freqspan, freqcenter)

#loc.plot_txdist_live()
# loc.calibrate(2)

# relative tx position
txpos = np.array([[0.0, 0.0],     # 433,91MHz
                  [80.0, 0.0],    # 434,16MHz
                  [40.0, 62.0]])  # 433,70MHz

x0 = np.array([30, 40])  # initial estimate



txpos = np.array([[0.0, 0.0],     # 433,91MHz
                  [80.0, 0.0]])  # 433,70MHz
#rf.get_measdata_from_file('measdata.txt', txpos)

#loc.map_path_ekf(x0, False, False, True)
