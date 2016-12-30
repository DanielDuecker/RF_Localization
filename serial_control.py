import time
import serial


class motor_communication(object):

    def __init__(self, portname, name):  #
        self.__oserial = []
        self.__portname = portname
        self.__name = name
        self.__isopen = False
        self.__timewritewait = 0.2
        self.__timereadwait = 0.2
        self.__signal = []
        self.__signallist = ['p', 'h', 'f']
        self.__ismoving = False
        self.__tempval = []
        self.__posinc = []
        self.__tposinc = []
        self.__posmm = []
        self.__tposmm = []
        self.__rpm = []

        self.reset_signal()

    def open_port(self):
        """
        :return:
        """
        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.__oserial = serial.Serial(
            port=self.__portname,
            baudrate=9600,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        self.__oserial.isOpen()  # open serial port
        self.__isopen = True
        print('Serial port ' + self.__portname + ' is open!')
        return True

    def close_port(self):
        self.__oserial.close()
        self.__isopen = False
        print('Serial port ' + self.__portname + ' is open!')
        return True

    def reset_signal(self):
        self.__signal = 0

    def listen_to_port(self, waitingfortype='rpm'):

        out = ''
        time.sleep(self.__timereadwait)

        #while self.__oserial.inWaiting() > 0:
        #    new_data = self.__oserial.read(1)
        #    out += new_data  # pure number string

        teststring = '-2000\r\np\r\nf\r\nOK\r\n'

        out = teststring
        out_split = out.rstrip().split('\r\n')
        for item in out_split:
            try:
                self.__tempval = int(item)
                print ('numberfound')
            except ValueError:
                if item == 'p':
                    self.__signal = item
                    print ('p found')
                elif item == 'h':
                    self.__signal = item
                    print ('h found')
                elif item == 'f':
                    self.__signal = item
                    print ('f found')
                else:
                    print('Unknown signal found on serial port: "' + item + '"')
        print(out_split)
        return True  # pure number string

    def write_on_port(self, strcommand):
        self.__oserial.write(strcommand + '\r\n')
        time.sleep(self.__timewritewait)
        return True

    def get_rpm(self):
        self.write_on_port('GN')
        self.listen_to_port('rpm')
        if abs(self.__tempval) < 10000:  # max motor speed = 7000rpm
            self.__rpm = self.__tempval
        return self.__rpm

    def get_pos(self):
        self.__oserial.write('POS' + '\r\n')
        time.sleep(0.2)
        self.listen_to_port('pos')
        self.__posinc = self.__tempval
        return self.__posinc

    def check_moving(self):
        if abs(self.get_rpm()) < 10:
            self.__ismoving = False
            return False
        else:
            self.__ismoving = True
            return True

    def check_arrival(self):
        if self.__signal == 'p':
            return True
        elif self.__signal in 'fh':
            self.get_pos()
            print('ERROR: ' + self.__name + ' reached limit of travel at pos ' + str(self.__posinc) + ' !')
        self.reset_signal()

    def get_status(self):
        print('Portname: ' + self.__portname)
        print('Name: ' + self.__name)
        print('Port open: ' + str(self.__isopen))
        print('Time wait after writing: ' + str(self.__timewritewait))
        print('Time wait before reading: ' + str(self.__timereadwait))
        print('Signal' + str(self.__signal))
        print('Signallist' + str(self.__signallist))
        print('IsMoving: ' + str(self.__ismoving))
        print('TempVal = ' + str(self.__tempval))
        print('PosInc = ' + str(self.__posinc))
        print('TPosInc = ' + str(self.__tposinc))
        print('RPM = ' + str(self.__rpm))

    def go_to_posinc(self, tposinc):
        moving_seq = ['LA'+str(tposinc),  # set absolute target position in [inc]
                      'NP',  # activate 'NotifyPosition' --> sends 'p' if position is reached
                      'M']  # start motion
        trying = True
        counter = 0
        maxtrys = 10
        while trying:
            counter += 1
            self.check_moving()
            if self.__ismoving is False:
                for command in moving_seq:
                    self.write_on_port(command)

                trying = False
                self.check_moving()
                print ('Start moving to Position: ' + str(tposinc))
            else:
                time_wait = 1.0
                print(self.__name + ' cannot move to new target position, still moving to old target position!')
                print('Waiting for ' + str(time_wait) + 's')
                print('Try ' + str(counter) + '/' + str(maxtrys))
                time.sleep(time_wait)
            if counter >= maxtrys:
                trying = False  # give up trying

        return True


    def initialize_home_pos(self):

        if self.check_moving() is False:
            self.write_on_port('GOHOSEQ')
            if self.check_moving() is True:
                cominghome = True
                while cominghome:
                    # give position
                    # check arrival
                    time.sleep(1.0)
                    # display speed
            else:
                # find out some sequence v1000 / -v1000 etc...
        else:
            print('Cannot start homing sequence ' + self.__name + ' is moving!')











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
        return isnumber, out.split('\r\n')  # pure number string

    def found_arrival_mark(out):
        barrived = False
        if out == 'p':
            print('arrived at target position')
            barrived = True
        elif out == 'h':
            print('arrived at home position')
            barrived = True
        elif out == 'f':
            print('arrived at fault position')
            barrived = True
        else:
            print('ERROR in "found_arrival_mark": this must not happen!')
            barrived = False
        return barrived  # has stopped moving


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
            'V0'
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
                    else:
                        bmoving = found_arrival_mark(str_rpm)  # arrival marker = stop mark
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