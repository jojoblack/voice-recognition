from tkinter import *
from tkinter import ttk
import tkinter.filedialog as filedialog

def selectFile():
    filename.set(filedialog.askopenfilename(filetypes=(("All files", "*.*"), )))

def recognize():
    #result = beginRecognize(filename)
    #result = filename.get()
    #showResult(result)
    result.set(filename.get());
    
def showResult(result):
    top = Toplevel()
    top.geometry('30x30')
    label = ttk.Label(top, text=result).pack()
        
root = Tk()
root.title("声纹识别")
root.geometry("680x200")

mainframe = ttk.Frame(root, padding="2 4 20 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

global filename
filename = StringVar()
result = StringVar()

ttk.Entry(mainframe, width=80, text=filename).grid(column=2, row=1, sticky=(W, E))

ttk.Button(mainframe, text="选择文件", command=selectFile).grid(column=1, row=1, sticky=W)
ttk.Button(mainframe, text="进行识别", command=recognize).grid(column=1, row=2, sticky=W)
ttk.Label(mainframe, text="识别结果：  ").grid(column=1, row = 3, sticky = E)
ttk.Entry(mainframe, width=80, text=result).grid(column=2, row=3, sticky=(W, E))
#ttk.Label(mainframe, text="正 确 率：  ").grid(column=1, row = 4, sticky = E)
#ttk.Entry(mainframe, width=80, text=accuracy).grid(column=2, row=4, sticky=(W, E))



for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)


root.mainloop()