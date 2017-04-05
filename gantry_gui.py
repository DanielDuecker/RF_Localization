import Tkinter as Tk
import ttk
import gantry_control
import rf_tools

LARGE_FONT = ('Verdana', 12)


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
        self.__gt = gantry_control.GantryControl([0, 3100, 0, 1600], use_gui)
        oBelt = self.__gt.get_serial_x_handle()
        oSpindle = self.__gt.get_serial_y_handle()

        label_pos_xy = ttk.Label(self, text='X = ?mm\nY = ?mm')
        label_pos_xy.grid(row=1, column=1)

        def get_position():
            pos_x_mm, pos_y_mm = self.__gt.get_gantry_pos_xy_mm()
            label_pos_xy.configure(text='X = ' + str(int(pos_x_mm)) + 'mm \nY = ' + str(int(pos_y_mm)) + 'mm')
            return True

        button_gantry_position = ttk.Button(self, text='Update Position', command=lambda: get_position())
        button_gantry_position.grid(row=1, column=0)

        """
        Belt-Drive
        """
        firstrow_belt = 2

        label_spindle_name = ttk.Label(self, text='Belt-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=firstrow_belt + 0, column=1)

        button3 = ttk.Button(self, text='v-- V [-]',
                             command=lambda: oBelt.set_drive_speed(-1 * int(entry_v_belt.get())))
        button3.grid(row=firstrow_belt + 1, column=0)

        button4 = ttk.Button(self, text='STOP', command=lambda: oBelt.set_drive_speed(0))
        button4.grid(row=firstrow_belt + 1, column=1)

        button5 = ttk.Button(self, text='[+] V --^', command=lambda: oBelt.set_drive_speed(1 * int(entry_v_belt.get())))
        button5.grid(row=firstrow_belt + 1, column=2)

        label_v_belt = ttk.Label(self, text='Velocity:')
        entry_v_belt = ttk.Entry(self)
        entry_v_belt.insert(0, '0')

        label_v_belt.grid(row=firstrow_belt + 2, column=0)
        entry_v_belt.grid(row=firstrow_belt + 2, column=1)

        button_manual_mode_belt = ttk.Button(self, text=' Manual Mode Belt', command=lambda: oBelt.start_manual_mode())
        button_manual_mode_belt.grid(row=firstrow_belt + 2, column=3)

        """
        Spindle-Drive
        """
        firstrow_spindle = 5

        label_spindle_name = ttk.Label(self, text='Spindle-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=firstrow_spindle+0, column=1)

        button2 = ttk.Button(self, text='<-- V [-]', command=lambda: oSpindle.set_drive_speed(-1*int(entry_v_spindle.get())))
        button2.grid(row=firstrow_spindle+1, column=0)

        button3 = ttk.Button(self, text='STOP', command=lambda: oSpindle.set_drive_speed(0))
        button3.grid(row=firstrow_spindle+1, column=1)

        button4 = ttk.Button(self, text='[+] V -->', command=lambda: oSpindle.set_drive_speed(1*int(entry_v_spindle.get())))
        button4.grid(row=firstrow_spindle+1, column=2)

        label_v_spindle = ttk.Label(self, text='Velocity:')
        entry_v_spindle = ttk.Entry(self)
        entry_v_spindle.insert(0, '0')

        label_v_spindle.grid(row=firstrow_spindle+2, column=0)
        entry_v_spindle.grid(row=firstrow_spindle+2, column=1)

        button_manual_mode_spindle = ttk.Button(self, text=' Manual Mode Spindle', command=lambda: oSpindle.start_manual_mode())
        button_manual_mode_spindle.grid(row=firstrow_spindle+2, column=3)

        """
        Quit-Button
        """
        button_home_seq = ttk.Button(self, text='Initialize Home Position', command=lambda: self.__gt.start_go_home_seq_xy())
        button_home_seq.grid(row=8, column=3, sticky='W', pady=4)

        button_quit = ttk.Button(self, text='Quit', command=self.quit)
        button_quit.grid(row=8, column=4, sticky='W', pady=4)

        label = ttk.Label(self, text='Start Page')
        label.grid(row=8, column=0)

        button1 = ttk.Button(self, text='Drive Settings', command=lambda: controller.show_frame(PageOne))
        button1.grid(row=8, column=1)

        button_start_field_meas = ttk.Button(self, text='Start EM-Field Measurement',
                                             command=lambda: self.__gt.start_field_measurement_file_select())
        button_start_field_meas.grid(row=8, column=5, sticky='W', pady=4)

        #button_analyze_data = ttk.Button(self, text='Analyze Data',
        #                             command=lambda: rf_tools.analyze_measdata_from_file())
        #button_analyze_data.grid(row=8, column=6, sticky='W', pady=4)


class PageOne(Tk.Frame):
    def __init__(self, parent, controller):
        Tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text='Drive Settings', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text='Controller Menu', command=lambda: controller.show_frame(StartPage))
        button1.pack()

app = GantryGui()

app.mainloop()

