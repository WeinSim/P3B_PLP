import math
import random as rd
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from expressions import *

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

    d = np.arange(0.5e-3, 6.25e-3, 0.25e-3)
    p = [ 4.0 if i < 8 else 4.1 for i in range(1, 24) ]
    u = [ 655, 671.0, 686, 487.3, 494.7, 505.4, 430.9, 433.4, 431.1, 405.6, 398.8, 395.8, 356.0, 356.6, 373.8, 359.2, 347.4, 347.9, 344.9, 345.0, 344.7,
          356.0, 349.0, 348.1, 351.6, 350.7, 349.7, 356.0, 364.7, 358.4, 357.4, 355.4, 356.0, 361.2, 363.3, 363.2, 373.5, 373.1, 374.3, 371.2, 370.9,
          375.4, 370.5, 372.4, 372.5, 371.5, 370.2, 368.8, 367.3, 370.2, 370.8, 380.3, 375.0, 347.3, 368.8, 372.4, 383.6, 395.5, 397.0, 388.8, 416.7,
          386.0, 384.6, 404.7, 402.9, 401.0, 393.3, 401.2, 405.8 ]

    dp = [0] * (3 * d.size)
    for i in range(d.size):
        dp[3 * i] = d[i] * p[i] * u[3 * i]
        dp[3 * i + 1] = d[i] * p[i] * u[3 * i + 1]
        dp[3 * i + 2] = d[i] * p[i] * u[3 * i + 2]
     # dp = [d_i[math.floor(i / 3)] * p[ for (d_i, p_i) in zip(d, p)]

    # fit curves to data
    fit_functions = [paschen, paschen, f]
    start_params = [None, (80, 2, 0.7), (80, 2, 0.7)]
    for i in range(3):
        fit_function = fit_functions[i]
        start_param = start_params[i]
        
        print(f"Funktion = {"Paschen" if fit_function == paschen else "f"}, Startwerte = {start_param}")
        params, cov = curve_fit(fit_function, dp, u, start_param)
        # params = default_params
        uncertainties = np.sqrt(np.diag(cov))
        # params = default_params
        print("Fitparameter (A, B, C):")
        print("Werte:          " + "".join(f"{p:10.4g}" for p in params))
        print("Unsicherheiten: " + "".join(f"{u:10.4g}" for u in uncertainties))

        (A, B, C) = params

        dprange = np.linspace(min(dp), max(dp), 100)
        fit = fit_function(dprange, A, B, C)

        pp = PdfPages(f"../Abbildungen/Graph_TV1_{i + 1}.pdf")
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

        if not i == 2:
            print()
            continue
 
        A_var = Var(A, uncertainties[0], "A")
        B_var = Var(B, uncertainties[1], "B")
        C_var = Var(C, uncertainties[2], "C")
        u_min_expr = Div(Mult(A_var, Const(math.e)), B_var)
        u_min = u_min_expr.eval()
        d_u_min = gaussian(u_min_expr, [A_var, B_var, C_var])
        print(f"U_min = {u_min}")
        print(f"\u0394U_min = {d_u_min}")


def paschen(x, A, B, C):
    return B * x / (np.log(A * x) - C)

def tv3():
    print("--- Teilversuch 3 ---")

    r = [ 0.008, 0.021, 0.038, 0.062, 0.082, 0.104, 0.138, 0.163, 0.185, 0.208 ]
    u = [ 16, 12, 8.8, 6.4, 4.8, 3.8, 2.4, 2.0, 1.5, 1.1 ]

    log_r = [math.log(r_i / r[0]) for r_i in r]
    log_u = [math.log(u_i / u[0]) for u_i in u]

    params, cov = curve_fit(linear_function, log_r, log_u)
    uncertainties = np.sqrt(np.diag(cov))
    print("Fitparameter (m, b):")
    print("Werte:          " + "".join(f"{p:10.4g}" for p in params))
    print("Unsicherheiten: " + "".join(f"{u:10.4g}" for u in uncertainties))

    (m, b) = params
    rrange = np.linspace(min(log_r), max(log_r), 100)
    fit = linear_function(rrange, m, b)

    pp = PdfPages(f"../Abbildungen/Graph_TV3.pdf")
    fig = plt.figure()

    plt.plot(log_r, log_u, "o", label="Messwerte")
    plt.plot(rrange, fit, "-", label=f"Ausgleichsgerade. m = {m:.2f}, b = {b:.2f}")

    plt.xlabel("$log($Abstand$)$")
    plt.ylabel("$log($Spannung$)$")
    # plt.yscale("log")
    # plt.title("Zündspannung vs. Abstand * Druck")
    plt.legend()

    fig.tight_layout()
    pp.savefig()
    pp.close()

def linear_function(x, m, b):
    return m * x + b

vorbereitung()
tv1()
tv3()
