import time
import serial


def test_serial():

    def listen_port(serport):
        out = ''
        isnumber = True
        while serport.inWaiting() > 0:
            new_data = serport.read(1)

            if new_data == 'p':
                isnumber = False
                out = new_data
                return isnumber, out
            elif new_data == 'h':
                isnumber = False
                out = new_data
                return isnumber, out
            elif new_data == 'f':
                isnumber = False
                out = new_data
                return isnumber, out
            else:
                out += new_data  # pure number string
        return isnumber, out  # pure number string


    def get_rpm(serport):
        input = 'GN'
        serport.write(input + '\r\n')
        time.sleep(0.5)

        [isnumber, out] = listen_port(serport)

        return isnumber, out

    def get_pos(serport):
        input = 'POS'
        serport.write(input + '\r\n')
        time.sleep(0.5)

        [isnumber, out] = listen_port(serport)

        return isnumber, out

    def go_to_wp(serport, wp_inc):
        bmoving = False
        readyforinput = True

        commands = [
            'LA'+str(wp_inc),
            'NP',
            'M']
        step = 0
        running = True
        while running:
            if readyforinput:

                if step >= len(commands):
                    print('reached end of command list')
                    running = False  # leave loop

                else:
                    print('commline: ' + str(step))
                    input = commands[step]
                    step += 1  # next line

                    print('Execute command: >>' + input)
                    # send the character to the device
                    # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
                    serport.write(input + '\r\n')

            out = ''
            # let's wait 0.2 second before reading output (let's give device time to answer)
            time.sleep(0.2)

            while serport.inWaiting() > 0:
                out += serport.read(1)

            if out != '':
                print ("<<" + out)
                if out == 'OK\r\n':
                    print ('confirmed command')
                    time.sleep(0.5)
                    isnumber, str_rpm = get_rpm(serport)
                    if isnumber:
                        print('rpm: ' + str_rpm)
                        int_rpm = int(str_rpm)
                        if abs(int_rpm) > 10:
                            print('gantry is moving')
                            readyforinput = False
                            bmoving = True
        return bmoving

    def check_arrival(serport):
        # let's wait 0.2 second before reading output (let's give device time to answer)
        time.sleep(0.2)

        isnumber, out = listen_port(serport)
        if isnumber is False and out != '':
            print ("<<" + out)
            if out == 'p':
                print('arrived at target position')
            elif out == 'h':
                print('arrived at home position')
            elif out == 'f':
                print('arrived at fault position')
            return True  # has stopped moving
        else:
            return False



    def test_mod():
        running = True
        commline = 0

        tpos1 = 1500000
        tpos2 = 1000000

        commands = [
            'LA' + str(tpos1),  # position 1
            'NP',
            'M',
            'LA' + str(tpos2),  # position 2
            'NP',
            'M',
            'LA' + str(tpos1),  # position 1
            'NP',
            'M']
        print ('Enter your commands below.\r\n'
               'chose gantry mode by inserting "auto" or "manual".')
        rawinput = raw_input(">> ")
        if rawinput == 'auto':
            gantry_mode = 'auto'
        elif rawinput == 'manual':
            gantry_mode = 'manual'

        readyforinput = True
        while running:
            if readyforinput:
                # get keyboard input

                # Python 3 users
                # input = input(">> ")
                if gantry_mode == 'auto':
                    if commline >= len(commands):
                        print('reached end of command list')
                        running = False  # leave loop

                    else:
                        print('commline: ' + str(commline))
                        input = commands[commline]
                        commline += 1  # next line

                        print('Execute command: >>' + input)
                        # send the character to the device
                        # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
                        ser3.write(input + '\r\n')
                elif gantry_mode == 'manual':
                    rawinput = raw_input(">> ")
                    if rawinput == 'exit':
                        running = False

            out = ''
            # let's wait 0.2 second before reading output (let's give device time to answer)
            time.sleep(0.2)

            while ser3.inWaiting() > 0:
                out += ser3.read(1)

            if out != '':
                print ("<<" + out)
                if out == 'OK\r\n':
                    print ('confirmed command')
                    time.sleep(0.5)
                    str_rpm = get_rpm(ser3)
                    print('rpm: ' + str_rpm)
                    if abs(int(str_rpm)) > 10:
                        print('gantry is moving')
                        readyforinput = False

                elif out == 'p\r\n':
                    print('arrived at position')
                    readyforinput = True
                elif out == 'h\r\n':
                    print('arrived at home position')
                    readyforinput = True
                elif out == 'f\r\n':
                    print('arrived at fault position')
                    readyforinput = True

                    # str_pos = get_pos(ser3)
                    # print ('act_pos: ' + str_pos)


                    # act_pos = get_actual_pos(ser3)
                    # print('Actual position: ' + act_pos)





    # configure the serial connections (the parameters differs on the device you are connecting to)
    ser3 = serial.Serial(
        port='/dev/ttyS4',   # s4 -> laengs = com3  # s5 -> quer = com4
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )

    ser3.isOpen()

    wp_list = [
        10000,
        100000,
        200000,
        300000,
        400000,
        500000,
        600000,
        700000,
        800000]

    for i in range(len(wp_list)):
        print('wp: ' + str(wp_list[i]))
        bgantry_moving = go_to_wp(ser3, wp_list[i])

        while bgantry_moving:
            time.sleep(0.5)
            arrival_type = check_arrival(ser3)
            if arrival_type == 'p\r\n':
                bgantry_moving = False
            else:
                print('Gantry has not arrived yet')

    print('exit and close serial')
    ser3.close()
    exit()





def serial_example():
    # configure the serial connections (the parameters differs on the device you are connecting to)
    ser3 = serial.Serial(
        port='/dev/ttyS4',  # s4 -> laengs = com3  # s5 -> quer = com4
        baudrate=9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS
    )

    ser3.isOpen()

    print 'Enter your commands below.\r\nInsert "exit" to leave the application.'
    running = True
    input = 1
    while running:
        # get keyboard input
        input = raw_input(">> ")
        # Python 3 users
        # input = input(">> ")

        if input == 'exit':
            ser3.close()
            exit()
        else:
            # send the character to the device
            # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
            ser3.write(input + '\r\n')
            out = ''
            # let's wait 0.2 second before reading output (let's give device time to answer)
            time.sleep(.2)
            while ser3.inWaiting() > 0:
                out += ser3.read(1)

            if out != '':
                print "<<" + out