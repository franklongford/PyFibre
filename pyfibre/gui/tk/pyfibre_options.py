from tkinter import (
    Frame, Toplevel, Label, HORIZONTAL, Scale, N, W, E, S,
    Checkbutton
)


class PyFibreOptions:

    def __init__(self, parent, width=320, height=620):

        self.parent = parent
        self.width = width
        self.height = height

        "Initialise option parameters"
        self.window = Toplevel(self.parent.master)
        self.window.tk.call('wm', 'iconphoto', self.window._w, self.parent.title.image)
        self.window.title('PyFibre - Options')
        self.window.geometry(f"{width}x{height}-100+40")
        self.window.configure(background='#d8baa9')

        self.frame = Frame(self.window)
        self.create_options()

    def create_options(self):

        self.title_sigma = Label(self.frame, text="Gaussian Std Dev (pix)")
        self.title_sigma.configure(background='#d8baa9')
        self.sigma = Scale(self.frame, from_=0, to=10, tickinterval=1, resolution=0.1,
                length=300, orient=HORIZONTAL, variable=self.parent.sigma)

        self.title_sigma.grid(column=0, row=2, rowspan=1)
        self.sigma.grid(column=0, row=3, sticky=(N,W,E,S))

        self.title_p0 = Label(self.frame, text="Low Clip Intensity (%)")
        self.title_p0.configure(background='#d8baa9')
        self.p0 = Scale(self.frame, from_=0, to=100, tickinterval=10,
                length=300, orient=HORIZONTAL, variable=self.parent.p0)

        self.title_p1 = Label(self.frame, text="High Clip Intensity (%)")
        self.title_p1.configure(background='#d8baa9')
        self.p1 = Scale(self.frame, from_=0, to=100,tickinterval=10,
                length=300, orient=HORIZONTAL, variable=self.parent.p1)

        self.title_p0.grid(column=0, row=4, rowspan=1)
        self.p0.grid(column=0, row=5)
        self.title_p1.grid(column=0, row=6, rowspan=1)
        self.p1.grid(column=0, row=7)

        self.title_n = Label(self.frame, text="NL-Mean Neighbourhood 1 (pix)")
        self.title_n.configure(background='#d8baa9')
        self.n = Scale(self.frame, from_=0, to=100, tickinterval=10,
                length=300, orient=HORIZONTAL, variable=self.parent.n)

        self.title_m = Label(self.frame, text="NL-Mean Neighbourhood 2 (pix)")
        self.title_m.configure(background='#d8baa9')
        self.m = Scale(self.frame, from_=0, to=100,tickinterval=10,
                length=300, orient=HORIZONTAL, variable=self.parent.m)

        self.title_n.grid(column=0, row=8, rowspan=1)
        self.n.grid(column=0, row=9)
        self.title_m.grid(column=0, row=10, rowspan=1)
        self.m.grid(column=0, row=11)

        self.title_alpha = Label(self.frame, text="Alpha network coefficient")
        self.title_alpha.configure(background='#d8baa9')
        self.alpha = Scale(self.frame, from_=0, to=1, tickinterval=0.1, resolution=0.01,
                        length=300, orient=HORIZONTAL, variable=self.parent.alpha)

        self.title_alpha.grid(column=0, row=12, rowspan=1)
        self.alpha.grid(column=0, row=13)

        self.chk_metric = Checkbutton(self.frame, text="o/w metrics", variable=self.parent.ow_metric)
        self.chk_metric.configure(background='#d8baa9')
        self.chk_metric.grid(column=0, row=14, sticky=(N,W,E,S))

        self.chk_segment = Checkbutton(self.frame, text="o/w segment", variable=self.parent.ow_segment)
        self.chk_segment.configure(background='#d8baa9')
        self.chk_segment.grid(column=0, row=15, sticky=(N,W,E,S))

        self.chk_network = Checkbutton(self.frame, text="o/w network", variable=self.parent.ow_network)
        self.chk_network.configure(background='#d8baa9')
        self.chk_network.grid(column=0, row=16, sticky=(N,W,E,S))

        self.chk_figure = Checkbutton(self.frame, text="o/w figure", variable=self.parent.ow_figure)
        self.chk_figure.configure(background='#d8baa9')
        self.chk_figure.grid(column=0, row=17, sticky=(N,W,E,S))

        self.chk_db = Checkbutton(self.frame, text="Save Database", variable=self.parent.save_db)
        self.chk_db.configure(background='#d8baa9')
        self.chk_db.grid(column=0, row=18, sticky=(N,W,E,S))

        self.frame.configure(background='#d8baa9')
        self.frame.pack()
