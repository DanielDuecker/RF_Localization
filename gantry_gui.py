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

        self.__gt = gantry_control.GantryControl()

        oSpindle = self.__gt.get_serial_x_handle()
        oBelt = self.__gt.get_serial_x_handle()

        """
        Spindle-Drive
        """
        def set_spindle_velocity():
            oSpindle.set_target_speed_rpm(entry_v_spindle.get())
            return True

        label_spindle_name = ttk.Label(self, text='Spindle-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=1, column=1)

        button2 = ttk.Button(self, text='<-- V [-]', command=lambda: oSpindle.set_drive_speed(-oSpindle.get_target_speed_rpm()))
        button2.grid(row=2, column=0)

        button3 = ttk.Button(self, text='STOP', command=lambda: oSpindle.set_drive_speed(0))
        button3.grid(row=2, column=1)

        button4 = ttk.Button(self, text='[+] V -->', command=lambda: oSpindle.set_drive_speed(oSpindle.get_target_speed_rpm()))
        button4.grid(row=2, column=2)

        label_v_spindle = ttk.Label(self, text='Velocity:')
        entry_v_spindle = ttk.Entry(self)

        button_set_spindle_speed = ttk.Button(self, text='Set Speed', command=set_spindle_velocity)
        label_v_spindle.grid(row=3, column=0)
        entry_v_spindle.grid(row=3, column=1)
        button_set_spindle_speed.grid(row=3, column=2)

        """
        Belt-Drive
        """
        def set_belt_velocity():
            oBelt.set_target_speed_rpm(entry_v_belt.get())
            return True

        label_spindle_name = ttk.Label(self, text='Belt-drive', font=LARGE_FONT)
        label_spindle_name.grid(row=4, column=1)

        button3 = ttk.Button(self, text='v-- V [-]', command=lambda: oBelt.set_drive_speed(-oBelt.get_target_speed_rpm()))
        button3.grid(row=5, column=0)

        button4 = ttk.Button(self, text='STOP', command=lambda: oBelt.set_drive_speed(0))
        button4.grid(row=5, column=1)

        button5 = ttk.Button(self, text='[+] V --^', command=lambda: oBelt.set_drive_speed(-oBelt.get_target_speed_rpm()))
        button5.grid(row=5, column=2)

        label_v_belt = ttk.Label(self, text='Velocity:')
        entry_v_belt = ttk.Entry(self)
        button_set_belt_speed = ttk.Button(self, text='Set Speed', command=set_belt_velocity)
        label_v_belt.grid(row=6, column=0)
        entry_v_belt.grid(row=6, column=1)
        button_set_belt_speed.grid(row=6, column=2)

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
