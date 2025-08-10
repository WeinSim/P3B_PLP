import math
import random as rd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# from Expressions import *

# markers - linestyle - color

def vorbereitung():
    print("--- Vorbereitung ---")

    pp = PdfPages(f"../Abbildungen/Graph_Vorbereitung.pdf")
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    createGraphVorbereitung(ax[0], 1, 24.4, 0.16, -5.7)
    createGraphVorbereitung(ax[1], 2, 18.7, 0.13, -9.5)

    # plt.title(f"Radius^2 vs. Interfernzordnung")
    # plt.xlabel('Interferenzordnung')
    # plt.ylabel('Radius^2 (m^2)')
    # plt.legend()

    fig.tight_layout()

    pp.savefig()
    pp.close()

def createGraphVorbereitung(ax, i, A, B, C):
    x = np.arange(0, 100, 0.1)
    y = [f(x_i, A, B, C) for x_i in x]

    ax.plot(x, y, ".")
    ax.set_title(f"Graph {i}: A = {A}, B = {B}, C = {C}")

    print(C + 1 / B)

def f(x, A, B, C):
    return A * (x - C) / math.log(B * (x - C))

vorbereitung()
