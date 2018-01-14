import serial_control as sc
import numpy as np
import time

# sc.serial_example()
# sc.test_serial()

belt_drive = sc.MotorCommunication('/dev/ttyS0', 'belt_drive', 115200, 'belt', 3100, 2e6)
#def __init__(self, portname, name, baudrate, drivetype, travelling_distance_mm, extreme_pos_inc):  #

litposmm = np.linspace(0, 2500, 11)
print (str(litposmm))

belt_drive.open_port()
belt_drive.start_manual_mode()
belt_drive.initialize_home_pos()
belt_drive.initialize_extreme_pos()

for tposmm in litposmm:

    belt_drive.go_to_pos_mm(tposmm)
    barrived = False
    while barrived is False:
        time.sleep(0.5)
        barrived = belt_drive.check_arrival()

    time.sleep(2)

belt_drive.get_status()
time.sleep(1.0)
belt_drive.close_port()



