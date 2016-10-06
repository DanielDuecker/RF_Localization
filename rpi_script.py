"""This module is used on a raspberry pi to read
PSD values on a certein frequency
"""
rf.t.sleep(120)

import rf

EAR_PI = rf.RFear(435e6)

EAR_PI.get_psd(24)


