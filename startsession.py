import rf

cal = rf.CalEar(433.4e6)
#cal.plot_psd()

#cal.plot_multi_rss_live(433.91e6, 434.16e6, 2e4)
freqtx = 433.91e6
rss, var = cal.measure_rss_var(freqtx)

#cal.make_test()

alpha, xi = cal.get_model(rss, var)

print ('Tx freq : ' + str(freqtx/1e6) + 'MHz, alpha = ' + str(alpha) + ' xi = ' + str(xi))
