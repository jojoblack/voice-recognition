from tkinter import *
from tkinter import ttk
import tkinter.filedialog as filedialog

def selectFile():
    filename.set(filedialog.askopenfilename(filetypes=(("All files", "*.*"), )))

def selectDirectory():
    filedirectory.set(filedialog.askdirectory())
    
def recognize1():
    #people = recognizeSingle(filename.get())
    #result.set(people)
    result.set(filename.get())

def recognize2():
    #acc = recognizeBatch(filedirectory.get())
    #accuracy.set(str(acc))
    accuracy.set(str(1))

root = Tk()
root.title("声纹识别")
root.geometry("680x240")

mainframe = ttk.Frame(root, padding="2 6 20 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)


filename = StringVar()
filedirectory = StringVar()
result = StringVar()
accuracy = StringVar()

ttk.Button(mainframe, text="选择文件", command=selectFile).grid(column=1, row=1, sticky=W)
ttk.Entry(mainframe, width=80, text=filename).grid(column=2, row=1, sticky=(W, E))
ttk.Button(mainframe, text="逐个识别", command=recognize1).grid(column=1, row=2, sticky=W)
ttk.Label(mainframe, text="识别结果：  ").grid(column=1, row = 3, sticky = E)
ttk.Entry(mainframe, width=80, text=result).grid(column=2, row=3, sticky=(W, E))
ttk.Button(mainframe, text="选择文件夹", command=selectDirectory).grid(column=1, row=4, sticky=W)
ttk.Entry(mainframe, width=80, text=filedirectory).grid(column=2, row=4, sticky=(W, E))
ttk.Button(mainframe, text="批量识别", command=recognize2).grid(column=1, row=5, sticky=W)
ttk.Label(mainframe, text="正 确 率：  ").grid(column=1, row = 6, sticky = E)
ttk.Entry(mainframe, width=80, text=accuracy).grid(column=2, row=6, sticky=(W, E))


for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)


root.mainloop()