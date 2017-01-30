import Tkinter as Tk
import ttk
import gantry_control

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
        label = ttk.Label(self, text='Start Page', font=LARGE_FONT)
        label.grid(row=0, column=0)

        button1 = ttk.Button(self, text='Visit Page 1', command=lambda: controller.show_frame(PageOne))
        button1.grid(row=0, column=1)

        use_gui = True
        self.__gt = gantry_control.GantryControl([0, 3000, 0, 1580], use_gui)

        oBelt = self.__gt.get_serial_x_handle()
        oSpindle = self.__gt.get_serial_y_handle()

        """
        Belt-Drive
        """
        firstrow_belt = 1

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

        label_v_belt.grid(row=firstrow_belt + 2, column=0)
        entry_v_belt.grid(row=firstrow_belt + 2, column=1)

        button_manual_mode_belt = ttk.Button(self, text=' Manual Mode Belt', command=lambda: oBelt.start_manual_mode())
        button_manual_mode_belt.grid(row=firstrow_belt + 2, column=3)

        """
        Spindle-Drive
        """
        firstrow_spindle = 4

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

        label_v_spindle.grid(row=firstrow_spindle+2, column=0)
        entry_v_spindle.grid(row=firstrow_spindle+2, column=1)

        button_manual_mode_spindle = ttk.Button(self, text=' Manual Mode Spindle', command=lambda: oSpindle.start_manual_mode())
        button_manual_mode_spindle.grid(row=firstrow_spindle+2, column=3)

        """
        button_gohoseq_spindle = ttk.Button(self, text='GoHomeSeq')
        button_set_max_inc_pos_spindle = ttk.Button(self, text='Set max position [inc]')
        entry_max_pos_inc_spindle = ttk.Entry(self, color='r')
        label_inc_unit_spindle = ttk.Label(self, text='[inc]')
        button_set_max_mm_pos_spindle = ttk.Button(self, text='Set max position [mm]')
        entry_max_pos_mm_spindle = ttk.Entry(self, color='r')
        label_mm_unit_spindle = ttk.Label(self, text='[mm]')
        """


        """
        Quit-Button
        """
        Tk.Button(self, text='Quit', command=self.quit).grid(row=7, column=0, sticky='W', pady=4)


class PageOne(Tk.Frame):
    def __init__(self, parent, controller):
        Tk.Frame.__init__(self, parent)
        label = ttk.Label(self, text='Page One', font=LARGE_FONT)
        label.pack(pady=10, padx=10)

        button1 = ttk.Button(self, text='Back to home', command=lambda: controller.show_frame(StartPage))
        button1.pack()




app = GantryGui()

app.mainloop()

#root.mainloop()

"""
self.label_1 = Tk.Label(container, text='Manual Command Belt-Pos [inc]')
self.label_2 = Tk.Label(container, text='Manual Command Spindle-Pos [inc]')
self.entry_1 = Tk.Entry(container)
self.entry_2 = Tk.Entry(container)

self.label_1.grid(row=0)
self.label_2.grid(row=1)
self.entry_1.grid(row=0, column=1)
self.entry_2.grid(row=1, column=1)

self.SendButtonBelt = Tk.Button(container, text='Send2Belt', command=self.test)
self.SendButtonBelt.grid(row=0, column=2, sticky='w')

self.SendButtonSpindle = Tk.Button(container, text='Send2Spindle', command=self.test)
self.SendButtonSpindle.grid(row=1, column=2, sticky='w')

self.quitButton = Tk.Button(container, text='Quit', command=frame.quit)
self.quitButton.grid(row=3)

def test(self):
print('test')
"""
