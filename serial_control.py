import time
import serial


class MotorCommunication(object):

    def __init__(self, portname, name, drivetype, travelling_distance_mm, extreme_pos_inc):  #
        self.__oserial = []
        self.__portname = portname
        self.__name = name
        self.__drivetype = drivetype
        self.__travelling_distance_mm = float(travelling_distance_mm)
        self.__isopen = False
        self.__timewritewait = 0.005  # [s]
        self.__timereadwait = 0.005  # [s]
        self.__signal = []
        self.__signallist = ['p', 'h', 'f']
        self.__homeknown = False

        self.__extremeknown = True
        self.__posincmax = int(extreme_pos_inc)

        self.__manualinit = False
        self.__ismoving = False
        self.__tempval = []

        self.__posinc = []
        self.__tposinc = []
        self.__posmm = []
        self.__tposmm = []
        self.__rpm = []
        self.__target_speed_rpm = 0

        self.__rpmmax = []
        self.__findingspeed = []

        self.reset_signal()
        self.load_drive_data(drivetype)

    def open_port(self):
        """
        :return:
        """
        # configure the serial connections (the parameters differs on the device you are connecting to)
        self.__oserial = serial.Serial(
            port=self.__portname,
            baudrate=19200,
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
        print('Serial port ' + self.__portname + ' closed!')
        return True

    def reset_signal(self):
        self.__signal = 0

    def set_manual_init(self,bmanualinit):
        self.__manualinit = bmanualinit

    def get_manual_init(self):
        return self.__manualinit

    def listen_to_port(self, waitingfortype='rpm'):

        out = ''
        time.sleep(self.__timereadwait)

        while self.__oserial.inWaiting() > 0:
            new_data = self.__oserial.read(1)
            out += new_data  # pure number string

        # teststring = '-2000\r\np\r\nf\r\nOK\r\n'
        # out = teststring

        out_split = out.rstrip().split('\r\n')
        for item in out_split:
            try:
                self.__tempval = int(item)
                #print ('numberfound')
            except ValueError:
                if item == 'p':
                    self.__signal = item
                    # print ('Arrived at target position -> p-flag')
                elif item == 'h':
                    self.__signal = item
                    print ('h found')
                elif item == 'f':
                    self.__signal = item
                    print ('f found')
                elif item == 'OK':
                    dummy = 1
                    #print('Debugmessage: Ignore "OK"')
                else:
                    print('Unknown signal found on serial port: "' + item + '"')
        # print(out_split) # just for debugging
        return True  # pure number string

    def update_data(self):
        self.get_posinc()
        self.get_rpm()

    def load_drive_data(self, drivetype):
        if drivetype == 'belt':
            self.__rpmmax = 3000
            self.__findingspeed = 1000
        elif drivetype == 'spindle':
            self.__rpmmax = 7000
            self.__findingspeed = 2500
        else:
            print('Unknown drive type!')
            print('drive types known "belt" and "spindle"')
            print('exiting application')
            exit()

    def convert_mm2inc(self, pos_mm):
        pos_inc = int(pos_mm * (self.__posincmax/self.__travelling_distance_mm))
        return pos_inc

    def convert_inc2mm(self, pos_inc):
        pos_mm = pos_inc * (self.__travelling_distance_mm/self.__posincmax)
        return pos_mm

    def write_on_port(self, strcommand):
        self.__oserial.write(strcommand + '\r\n')
        time.sleep(self.__timewritewait)
        return True

    def set_home_pos_known(self, bknown):
        self.__homeknown = bknown

    def get_rpm(self):
        self.write_on_port('GN')
        self.listen_to_port('rpm')
        if abs(self.__tempval) < 10000:  # max motor speed = 7000rpm
            self.__rpm = self.__tempval
        return self.__rpm

    def set_posincmax(self, posincmax):
        self.__posincmax = posincmax

    def get_posinc(self):
        """
        gets the actual position and updates the member variables for position (both [inc] and [mm])
        :return: pos_inc [inc]
        """
        self.write_on_port('POS')
        self.listen_to_port('pos')
        self.__posinc = self.__tempval
        if self.__posincmax != []:
            self.__posmm = self.convert_inc2mm(self.__posinc)
        return self.__posinc

    def get_posmm(self):
        """
        gets the actual position and updates the member variables for position (both [inc] and [mm])
        :return: pos_mm [mm]
        """
        self.write_on_port('POS')
        self.listen_to_port('pos')
        self.__posinc = self.__tempval
        self.__posmm = self.convert_inc2mm(self.__posinc)
        return self.__posmm

    def set_target_speed_rpm(self, target_speed_rpm):
        self.__target_speed_rpm = target_speed_rpm

    def get_target_speed_rpm(self):
        return self.__target_speed_rpm

    def set_target_posinc(self, target_posinc):
        self.__tposinc = target_posinc

    def set_target_posmm(self, target_posmm):
        self.__tposmm = target_posmm

    def get_target_posmm(self):
        return self.__tposmmp

    def is_home_pos_known(self):
        return self.__homeknown

    def is_extreme_pos_known(self):
        return self.__extremeknown

    def start_home_seq(self):
        self.write_on_port('GOHOSEQ')

    def check_initialization_status(self, extreme_pos_mode=False):
        """
        Checks if home position is known and closes application otherwise
        :return:
        """
        if self.is_home_pos_known() is False:
            print('Home position is unknown!')
            print('exiting method')
            return False
        if self.is_extreme_pos_known() is False and extreme_pos_mode is False:
            print('Extreme position is unknown!')
            print('exiting method')
            return False
        return True

    def check_moving(self):
        if abs(self.get_rpm()) < 10:
            self.__ismoving = False
            return False
        else:
            self.__ismoving = True
            return True

    def get_dist_to(self, target_pos_mm):
        """
        Calculates the distance [mm] from the actual position to the target position
        :param target_pos_mm
        :return: target_pos_mm - drive_pos_mm
        """
        drive_pos_mm = self.get_posmm()
        return target_pos_mm - drive_pos_mm

    def check_arrival_signal(self):
        self.update_data()
        if self.__signal == 'p':
            self.reset_signal()
            return True

        elif self.__signal == 'h' or self.__signal == 'f':
            self.reset_signal()
            return True

        self.reset_signal()
        return False

    def get_status(self):
        self.update_data()

        print('\n  ### Status Report for ' + self.__name + ' ###')
        print('Drive type: ' + str(self.__drivetype))
        print('Portname: ' + self.__portname)
        print('Name: ' + self.__name)
        print('Port open: ' + str(self.__isopen))
        print('Time wait after writing: ' + str(self.__timewritewait))
        print('Time wait before reading: ' + str(self.__timereadwait))
        print('Signal: ' + str(self.__signal))
        print('IsMoving: ' + str(self.__ismoving))
        print('TempVal: ' + str(self.__tempval))
        print('PosIncMax: ' + str(self.__posincmax))
        print('TPosInc: ' + str(self.__tposinc))
        print('PosInc: ' + str(self.__posinc))
        print('PosMM: ' + str(self.__posmm))
        print('RPM: ' + str(self.__rpm))

    def enter_manual_init_data(self):
        print('Do you want to enter extreme position manually? (yes/no)')
        input = raw_input("")
        if input == 'yes':
            print('Enter extreme position in [inc]-units:')
            input = raw_input("")
            self.__posincmax = int(input)
            self.__homeknown = True
            self.__extremeknown = True
            self.set_manual_init(True)
        return True

    def start_manual_mode(self, safetycheck=True):
        print ('Enter your commands below.')
        print ('Type "AUTO_MODE" for switching to AUTO_MODE')
        print ('Type "status" for a status report')
        print ('Type "exit" to leave manual mode')
        print ('Type "exitall" to close the application')
        running = True

        while running:
            # get keyboard input
            input = raw_input(">> ")
            # Python 3 users
            # input = input(">> ")

            if input == 'AUTO_MODE':
                if safetycheck:
                    print('\n  ###  SAFETY_CHECK for ' + self.__name + ' ###  ')
                    print('Is everything ready to start? (yes/no)')
                    safety_input = raw_input(">> ")
                    if safety_input == 'yes':
                        return True
                    else:
                        print ('Safetycheck: FAILED!')
                        print ('Switching to "MANUAL_MODE"\n')
                        print ('Enter your commands below.')
                        print ('Type "AUTO_MODE" for switching to AUTO_MODE')
                        print ('Type "status" for a status report')
                        print ('Type "exit" to leave manual mode')
                        print ('Type "exitall" to close the application')

            elif input == 'exit':
                break

            elif input == 'exitall':
                self.__oserial.close()
                exit()

            elif input == 'status':
                self.get_status()

            else:
                # send the character to the device
                # (note that I happend a \r\n carriage return and line feed to the characters - this is requested by my device)
                self.__oserial.write(input + '\r\n')
                out = ''
                # let's wait 0.1 second before reading output (let's give device time to answer)
                time.sleep(0.1)
                while self.__oserial.inWaiting() > 0:
                    out += self.__oserial.read(1)

                if out != '':
                    print "<<" + out

    def set_drive_speed(self, v_inc):

        command = 'V'+str(v_inc)  # start moving with v_inc speed
        self.write_on_port(command)
        self.check_moving()

        return True


    def go_to_pos_mm(self, tposmm):
        """

        :param tposinc: absolute target position in [inc]
        :return:
        """
        if self.check_initialization_status() is False:
            return False

        tposinc = self.convert_mm2inc(tposmm)
        self.set_target_posinc(tposinc)
        self.set_target_posmm(tposmm)

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
                # print ('Start moving to Position: ' + str(tposinc))
                return True
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
        """

        :return:
        """
        print('Start initialization sequence --> going home')
        if self.check_moving() is False:

            self.start_home_seq()

            if self.check_moving() is True:
                cominghome = True
                while cominghome:
                    # give position

                    time.sleep(1.0)

                    print(self.__name + ' Coming-Home position ' + str(self.get_posinc()) +
                          'inc @ ' + str(self.get_rpm()) + 'rpm')
                    # check arrival at extreme position
                    if self.__signal == 'h' or self.__signal == 'f':
                        self.__homeknown = True
                        print(self.__name + ' reached home position!\n')
                        time.sleep(0.2)
                        cominghome = False
                    self.reset_signal()

            # elif self.__ismoving is False and try_coming_home == 0:
            # todo: switch t0 manual mode

            else:
                print('Failed to start "go home sequence"! ')
                print('Leaving >>AUTO_MODE<< ...')
                print('Entering >>MANUAL_MODE<< ...')
        else:
            print('Cannot start homing sequence ' + self.__name + ' is moving!')

        print('Finished coming home sequence!')

    def initialize_extreme_pos(self):
        if self.check_initialization_status(True) is False:
            return False

        print('Start finding extreme position sequence')
        if self.check_moving() is False:
            self.write_on_port('V'+str(self.__findingspeed))

            if self.check_moving() is True:
                findingextreme = True
                while findingextreme:
                    # give position

                    time.sleep(2.0)

                    print(self.__name + ' Init.Extreme position ' + str(self.get_posinc()) +
                          'inc @ ' + str(self.get_rpm()) + 'rpm')

                    # check arrival at extreme position
                    if self.__signal == 'h' or self.__signal == 'f':
                        self.__extremeknown = True
                        print(self.__name + ' reached extreme position!')
                        time.sleep(0.2)
                        self.set_posincmax(self.get_posinc())
                        print('Extreme position at ' + str(self.__posincmax) + ' inc.\n')

                        findingextreme = False
                    self.reset_signal()

            # elif self.__ismoving is False and try_coming_home == 0:
            # todo: switch t0 manual mode

            else:
                print('Failed to start "finding extreme sequence"! ')
                #print('Leaving >>AUTO_MODE<< ...')
                #print('Entering >>MANUAL_MODE<< ...')
                print('This is still a todo... =(')

            if self.__homeknown and self.__extremeknown:
                print('Finished initialization sequence!')
                print('Home and extreme positions are known!\n')
                print('Go back to home position')

                time.sleep(1)

                self.go_to_pos_mm(0)

                barrived = False
                while barrived is False:
                    time.sleep(0.5)
                    barrived = self.check_arrival_signal()
                print('Good to be home:')
                print('Arrived at home position')
                time.sleep(2)


        else:
            print('Cannot start finding extreme position sequence ' + self.__name + ' is moving!')

