from PIL import ImageTk, Image
from tkinter import Frame, Label, TOP


class Title(Frame):

    def __init__(self, master, pyfibre_dir):
        super(Title, self).__init__(master)

        image = Image.open(pyfibre_dir + '/img/icon.ico')
        image = image.resize((300, 200))
        image_tk = ImageTk.PhotoImage(image)

        self.master.tk.call(
            'wm', 'iconphoto', self.master._w, image_tk
        )
        self.text_title = Label(self, image=image_tk)
        self.image = image_tk
        self.text_title.pack(side=TOP, fill="both", expand="yes")