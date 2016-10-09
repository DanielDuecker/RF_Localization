import rf

cal = rf.CalEar(433.4e6)
#cal.plot_psd()

cal.plot_multi_rss_live(433.91e6, 434.16e6, 2e4)
