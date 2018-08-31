import Tkinter
import tkFileDialog
import tkSimpleDialog


# ======== "Save as" dialog:
def save_as_dialog(windowtitle='Save as...', myFormats=[('Text file', '*.txt')]):

    root = Tkinter.Tk()
    root.withdraw()  # get rid of the tk-app window in the background
    filename = tkFileDialog.asksaveasfilename(parent=root, filetypes=myFormats, title=windowtitle)
    if len(filename) > 0:
        print ('Now saving under ' + filename)
    return filename


# ======== Select a directory:
def select_directory():
    root = Tkinter.Tk()
    root.withdraw()  # get rid of the tk-app window in the background
    dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
    if len(dirname) > 0:
        print ('You chose ' + dirname)
    return dirname


# ======== Select a file for opening:
def select_file(myFormats=[('Text file', '*.txt')]):
    root = Tkinter.Tk()
    root.withdraw()  # get rid of the tk-app window in the background
    filename = tkFileDialog.askopenfilename(parent=root, filetypes=myFormats, title='Choose a file')
    if filename is not None:
        print ('You chose ' + filename)

    #    data = file.read()
    #    file.close()
    #    print "I got %d bytes from this file." % len(data)
    return filename


# ======== Writing a description for your file and pass it on:
def write_descrition():
    root = Tkinter.Tk()
    root.withdraw()  # get rid of the tk-app window in the background
    textinput = tkSimpleDialog.askstring("Description prompt", "enter your description")
    if textinput is not None:
        print ('You chose: ' + textinput)
    return textinput
