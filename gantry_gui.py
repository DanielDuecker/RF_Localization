import Tkinter as Tk
import ttk
import gantry_control
import rf_tools
import numpy as np

LARGE_FONT = ('Tahoma', 12)
SUPERLARGE_FONT = ('Tahoma', 50)
tx_2_analyse = [1]  # [1, 2, 3, 4, 5, 6]

rad_or_deg = True  # Rad := True ; Deg := False

class GantryGui(Tk.Tk):

    def __init__(self, *args, **kwargs):

        Tk.Tk.__init__(self, *args, **kwargs)

        Tk.Tk.wm_title(self, 'Gantry Control')

        container = Tk.Frame(self)
        container.pack()

        container.pack(side='top', fill='both', expand=True)

        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky='nsew')

        self.show_frame(StartPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()  # raise frame to the front


class StartPage(Tk.Frame):

    def __init__(self, parent, controller):
        Tk.Frame.__init__(self, parent)

        use_gui = True
        self.__gt = gantry_control.GantryControl([0, 3000, 0, 1580, 0, (2*np.pi+1e-6)], use_gui)
        oBelt = self.__gt.get_serial_x_handle()
        oSpindle = self.__gt.get_serial_y_handle()
        oShaft = self.__gt.get_serial_a_handle()


        # Notebook
        #notebook_label = ttk.Label(self, text="Control")
        #notebook_label.grid(row=3, column=2, pady=3)

        notebook_frame = ttk.Notebook(self)
        notebook_frame.grid(row=4, column=2, padx=30, pady=4)

        velcontrl_frame = ttk.Frame(notebook_frame)
        absposcontrl_frame = ttk.Frame(notebook_frame)
        relposcontrl_frame = ttk.Frame(notebook_frame)
        man_contrl_frame = ttk.Frame(notebook_frame)

        notebook_frame.add(velcontrl_frame, text="Velocity Control")
        notebook_frame.add(absposcontrl_frame, text="Abs Position Control")
        notebook_frame.add(relposcontrl_frame, text="Rel Position Control")
        notebook_frame.add(man_contrl_frame, text="Manual Control")

        if rad_or_deg:
            label_pos_xy = ttk.Label(self, text='X = ? mm\nY = ? mm\nA = ? rad')
        else:
            label_pos_xy = ttk.Label(self, text='X = ? mm\nY = ? mm\nA = ? deg')
        label_pos_xy.grid(row=1, column=1)

        def get_position():
            pos_x_mm, pos_y_mm, pos_a_rad = self.__gt.get_gantry_pos_xya_mmrad()
            if rad_or_deg:
                label_pos_xy.configure(text='X = ' + str(int(pos_x_mm)) + ' mm \nY = ' + str(int(pos_y_mm)) + ' mm \nA = ' + str(round(float(pos_a_rad), 4)) + ' rad')
            else:
                label_pos_xy.configure(text='X = ' + str(int(pos_x_mm)) + ' mm \nY = ' + str(int(pos_y_mm)) + ' mm \nA = ' + str(round((float(pos_a_rad)*180/np.pi), 4)) + ' deg')
            return True

        button_gantry_position = ttk.Button(self, text='Update Position', command=lambda: get_position())
        button_gantry_position.grid(row=1, column=0)

        """
        Belt-Drive
        """
        firstrow_belt = 2

        label_spindle_name = ttk.Label(velcontrl_frame, text='Belt-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=firstrow_belt + 0, column=1)

        button3 = ttk.Button(velcontrl_frame, text='v-- V [-]',
                             command=lambda: oBelt.set_drive_speed(-1 * int(entry_v_belt.get())))
        button3.grid(row=firstrow_belt + 1, column=0)

        button4 = ttk.Button(velcontrl_frame, text='STOP',command=lambda: oBelt.set_drive_speed(0))
        button4.grid(row=firstrow_belt + 1, column=1)

        button5 = ttk.Button(velcontrl_frame, text='[+] V --^', command=lambda: oBelt.set_drive_speed(1 * int(entry_v_belt.get())))
        button5.grid(row=firstrow_belt + 1, column=2)

        label_v_belt = ttk.Label(velcontrl_frame, text='Velocity:')
        entry_v_belt = ttk.Entry(velcontrl_frame)
        entry_v_belt.insert(0, '0')

        label_v_belt.grid(row=firstrow_belt + 2, column=0)
        entry_v_belt.grid(row=firstrow_belt + 2, column=1)




        """
        Spindle-Drive
        """
        firstrow_spindle = 5

        label_spindle_name = ttk.Label(velcontrl_frame, text='Spindle-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=firstrow_spindle+0, column=1)

        button2 = ttk.Button(velcontrl_frame, text='<-- V [-]', command=lambda: oSpindle.set_drive_speed(-1*int(entry_v_spindle.get())))
        button2.grid(row=firstrow_spindle+1, column=0)

        button3 = ttk.Button(velcontrl_frame, text='STOP', command=lambda: oSpindle.set_drive_speed(0))
        button3.grid(row=firstrow_spindle+1, column=1)

        button4 = ttk.Button(velcontrl_frame, text='[+] V -->', command=lambda: oSpindle.set_drive_speed(1*int(entry_v_spindle.get())))
        button4.grid(row=firstrow_spindle+1, column=2)

        label_v_spindle = ttk.Label(velcontrl_frame, text='Velocity:')
        entry_v_spindle = ttk.Entry(velcontrl_frame)
        entry_v_spindle.insert(0, '0')

        label_v_spindle.grid(row=firstrow_spindle+2, column=0)
        entry_v_spindle.grid(row=firstrow_spindle+2, column=1)





        """
        Shaft-Drive
        """
        firstrow_shaft = 8

        label_shaft_name = ttk.Label(velcontrl_frame, text='Shaft-drive', font=LARGE_FONT)
        label_shaft_name.grid(row=firstrow_shaft+0, column=1)

        button2 = ttk.Button(velcontrl_frame, text='<-- V [-]', command=lambda: oShaft.set_drive_speed(-1*int(entry_v_shaft.get())))
        button2.grid(row=firstrow_shaft+1, column=0)

        button3 = ttk.Button(velcontrl_frame, text='STOP', command=lambda: oShaft.set_drive_speed(0))
        button3.grid(row=firstrow_shaft+1, column=1)

        button4 = ttk.Button(velcontrl_frame, text='[+] V -->', command=lambda: oShaft.set_drive_speed(1*int(entry_v_shaft.get())))
        button4.grid(row=firstrow_shaft+1, column=2)

        label_v_shaft = ttk.Label(velcontrl_frame, text='Velocity:')
        entry_v_shaft = ttk.Entry(velcontrl_frame)
        entry_v_shaft.insert(0, '0')

        label_v_shaft.grid(row=firstrow_shaft+2, column=0)
        entry_v_shaft.grid(row=firstrow_shaft+2, column=1)





        """
        Abs Postion control
        """
        entry_abs_pos_belt = ttk.Entry(absposcontrl_frame)
        entry_abs_pos_belt.insert(0, '')
        entry_abs_pos_belt.grid(row=2, column=1)

        entry_abs_pos_spindle = ttk.Entry(absposcontrl_frame)
        entry_abs_pos_spindle.insert(0, '')
        entry_abs_pos_spindle.grid(row=3, column=1)

        entry_abs_pos_shaft = ttk.Entry(absposcontrl_frame)
        entry_abs_pos_shaft.insert(0, '')
        entry_abs_pos_shaft.grid(row=4, column=1)

        if rad_or_deg:
            button_goto_abs_pos = ttk.Button(absposcontrl_frame, text='go to X/Y/A - pos [mm]/[rad]', command=lambda: self.__gt.go_to_abs_pos(1*abs(int(entry_abs_pos_belt.get())), 1*abs(int(entry_abs_pos_spindle.get())), 1*abs(float(entry_abs_pos_shaft.get()))))
        else:
            button_goto_abs_pos = ttk.Button(absposcontrl_frame, text='go to X/Y/A - pos [mm]/[deg]', command=lambda: self.__gt.go_to_abs_pos(1*abs(int(entry_abs_pos_belt.get())), 1*abs(int(entry_abs_pos_spindle.get())), 1*abs(float(entry_abs_pos_shaft.get())*np.pi/180)))
        button_goto_abs_pos.grid(row=5, column=1, sticky='W', pady=4)


        """
        Rel Postion control
        """
        entry_rel_pos_belt = ttk.Entry(relposcontrl_frame)
        entry_rel_pos_belt.insert(0, '0')
        entry_rel_pos_belt.grid(row=2, column=1)

        entry_rel_pos_spindle = ttk.Entry(relposcontrl_frame)
        entry_rel_pos_spindle.insert(0, '0')
        entry_rel_pos_spindle.grid(row=3, column=1)

        entry_rel_pos_shaft = ttk.Entry(relposcontrl_frame)
        entry_rel_pos_shaft.insert(0, '0')
        entry_rel_pos_shaft.grid(row=4, column=1)

        if rad_or_deg:
            button_goto_rel_pos = ttk.Button(relposcontrl_frame, text='move by dx dy da [mm]/[rad]', command=lambda: self.__gt.go_to_rel_pos(1*int(entry_rel_pos_belt.get()), 1*int(entry_rel_pos_spindle.get()), 1*float(entry_rel_pos_shaft.get())))
        else:
            button_goto_rel_pos = ttk.Button(relposcontrl_frame, text='move by dx dy da [mm]/[deg]', command=lambda: self.__gt.go_to_rel_pos(1*int(entry_rel_pos_belt.get()), 1*int(entry_rel_pos_spindle.get()), 1*float(entry_rel_pos_shaft.get())*np.pi/180))
        button_goto_rel_pos.grid(row=5, column=1, sticky='W', pady=4)

        """
        Manual Control
        """
        button_manual_mode_belt = ttk.Button(man_contrl_frame, text=' Manual Mode Belt', command=lambda: oBelt.start_manual_mode())
        button_manual_mode_belt.grid(row=firstrow_belt + 2, column=3)
        button_manual_mode_spindle = ttk.Button(man_contrl_frame, text=' Manual Mode Spindle', command=lambda: oSpindle.start_manual_mode())
        button_manual_mode_spindle.grid(row=firstrow_spindle+2, column=3)
        button_manual_mode_shaft = ttk.Button(man_contrl_frame, text=' Manual Mode Shaft', command=lambda: oShaft.start_manual_mode())
        button_manual_mode_shaft.grid(row=firstrow_shaft+2, column=3)



        """
        EKF_Path Button
        """
        button_ekf_path = ttk.Button(self, text='EKF-Path (old)', command=lambda: self.__gt.follow_wp_and_take_measurements())
        button_ekf_path.grid(row=1, column=2)

        entry_num_plot_points = ttk.Entry(self)
        entry_num_plot_points.insert(0, '1000')
        entry_num_plot_points.grid(row=1, column=4)
        entry_log_lin_ekf = ttk.Entry(self)
        entry_log_lin_ekf.insert(0, 'log')
        entry_log_lin_ekf.grid(row=2, column=4)
        button_path = ttk.Button(self, text='WP-Path Following', command=lambda: self.__gt.follow_wp_path_opt_take_measurements(int(entry_num_plot_points.get()), entry_log_lin_ekf.get()))
        button_path.grid(row=1, column=3)

        """
        Settings
        """

        entry_max_speed_belt = ttk.Entry(self)
        entry_max_speed_belt.insert(0, '3000')
        entry_max_speed_belt.grid(row=3, column=6)
        button_max_speed_belt = ttk.Button(self, text='set max Speed Belt (<=3000!)', command=lambda: self.__gt.set_new_max_speed_x(1*abs(int(entry_max_speed_belt.get()))))
        button_max_speed_belt.grid(row=3, column=5, sticky='W', pady=4)

        entry_max_speed_spindle = ttk.Entry(self)
        entry_max_speed_spindle.insert(0, '9000')
        entry_max_speed_spindle.grid(row=4, column=6)
        button_max_speed_spindle = ttk.Button(self, text='set max Speed Spindle (<=9000!)', command=lambda: self.__gt.set_new_max_speed_y(1*abs(int(entry_max_speed_spindle.get()))))
        button_max_speed_spindle.grid(row=4, column=5, sticky='W', pady=4)

        entry_max_speed_shaft = ttk.Entry(self)
        entry_max_speed_shaft.insert(0, '123')
        entry_max_speed_shaft.grid(row=5, column=6)
        button_max_speed_shaft = ttk.Button(self, text='set max Speed Shaft (<=123!)', command=lambda: self.__gt.set_new_max_speed_y(1*abs(int(entry_max_speed_shaft.get()))))
        button_max_speed_shaft.grid(row=5, column=5, sticky='W', pady=4)

        button_home_seq = ttk.Button(self, text='Initialize Home Position', command=lambda: self.__gt.start_go_home_seq_xya())
        button_home_seq.grid(row=15, column=3, sticky='W', pady=4)

        """
        Quit-Button
        """
        button_quit = ttk.Button(self, text='Quit', command=self.quit)
        button_quit.grid(row=15, column=4, sticky='W', pady=4)

        label = ttk.Label(self, text='Start Page')
        label.grid(row=15, column=0)

        button1 = ttk.Button(self, text='Drive Settings', command=lambda: controller.show_frame(PageOne))
        button1.grid(row=15, column=1)

        button_start_field_meas = ttk.Button(self, text='Start EM-Field Measurement',
                                             command=lambda: self.__gt.start_field_measurement_file_select())
        button_start_field_meas.grid(row=8, column=5, sticky='W', pady=4)

        entry_log_lin_analyze = ttk.Entry(self)
        entry_log_lin_analyze.insert(0, 'log')
        entry_log_lin_analyze.grid(row=8, column=7)
<<<<<<< HEAD
        button_analyze_data = ttk.Button(self, text='Analyze Data', command=lambda: rf_tools.analyze_measdata_from_file(entry_log_lin_analyze.get(), analyze_tx=tx_2_analyse))
=======
        button_analyze_data = ttk.Button(self, text='Analyze Data', command=lambda: rf_tools.analyze_measdata_from_file(entry_log_lin_analyze.get(), analyze_tx=[1, 2, 3, 4, 5, 6]))
>>>>>>> 62b35bf3963ab8a3daea5a51e3f0084498cd322b
        button_analyze_data.grid(row=8, column=6, sticky='W', pady=4)

        """
        Emergency-Stop-Button
        """
        button_stop = Tk.Button(self, text='STOP', width=6, command=(oBelt.set_drive_speed(0), oSpindle.set_drive_speed(0), oShaft.set_drive_speed(0)), background='#ff7070', activebackground='red', font=SUPERLARGE_FONT)
        button_stop.grid(row=4, column=1, sticky='W', pady=4, ipady=30)


class PageOne(Tk.Frame):
    def __init__(self, parent, controller):
        Tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text='Drive Settings', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text='Controller Menu', command=lambda: controller.show_frame(StartPage))
        button1.pack()


app = GantryGui()


app.mainloop()

