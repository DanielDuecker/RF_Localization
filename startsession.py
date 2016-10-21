import rf

cal = rf.CalEar(433.9e6)
#cal.plot_psd()

# cal.plot_multi_rss_live(433.91e6, 434.16e6, 2e4)
freqtx = 433.91e6
cal.make_test(5.0)
#rss, var = cal.measure_rss_var(freqtx, 2e4, 10.0)
#alpha, xi = cal.get_model(rss, var)

#print ('Tx freq : ' + str(freqtx/1e6) + 'MHz, alpha = ' + str(alpha) + ' xi = ' + str(xi))

#alpha = [0.148931742646]#, 0.148931742646]
#xi = [-216954385.393]#, -216954385.393]
#freqtx = [433.91e6]#, 433.91e6]
#freqspan = 2e5
#freqcenter = 433.9e6
#loc = rf.LocEar(alpha, xi, freqtx, freqspan, freqcenter)
##loc.calibrate()
#loc.map_path_multi_tx(80.0)
