import logging
from tkinter import Tk

from pyfibre.gui.pyfibre_gui import PyFibreGUI
from pyfibre.utilities import logo

logger = logging.getLogger(__name__)


def run():

    N_PROC = 1#os.cpu_count() - 1

    logger.info(logo())

    root = Tk()
    GUI = PyFibreGUI(root, N_PROC)

    root.mainloop()
