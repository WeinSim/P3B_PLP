import math
import random as rd
import numpy as np
from scipy.optimize import curve_fit

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
    return A * (x - C) / np.log(B * (x - C))

def tv1():
    print("--- Teilversuch 1 ---")

    d = [.001, .002, .003, .004, .005, .006]
    p = [2e2, 2e2, 2e2, 2e2, 2e2, 2e2]
    u = [750, 300, 200, 250, 350, 450]

    dp = [d_i * p_i for (d_i, p_i) in zip(d, p)]

    # fit paschen-curve to data
    default_params = (600, 8, 0.2)
    params, cov = curve_fit(paschen, dp, u, default_params)
    # params = default_params
    print("Fitparameter (A, B, C):")
    print(params)

    (A, B, C) = params
    u_min = A / B * np.exp(C + 1)
    print(f"U_min = {u_min}")

    dprange = np.linspace(min(dp), max(dp), 100)
    fit = paschen(dprange, A, B, C)

    pp = PdfPages(f"../Abbildungen/Graph_TV1.pdf")
    fig = plt.figure()

    plt.plot(dp, u, "o", label="Messwerte")
    plt.plot(dprange, fit, "-", label=f"Fit. A = {A:.2f}, B = {B:.2f}, C = {C:.2f}")

    plt.xlabel("Abstand * Druck [$m \\cdot Pa$]")
    plt.ylabel("Spannung [$V$]")
    # plt.title("Zündspannung vs. Abstand * Druck")
    plt.legend()

    fig.tight_layout()
    pp.savefig()
    pp.close()

def paschen(x, A, B, C):
    return A * x / (np.log(B * x) - C)

# vorbereitung()
tv1()
