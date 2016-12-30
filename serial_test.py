import serial_control as sc

# sc.serial_example()
# sc.test_serial()

motor1 = sc.motor_communication('huhu', 'motor1')
motor1.listen_to_port()
motor1.check_moving()
motor1.check_arrival()

