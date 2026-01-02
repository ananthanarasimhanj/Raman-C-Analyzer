import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# ========= SETTINGS =========
LASER_NM = 532
CARBON_RANGE = (1000, 3200)
# ============================

def lorentz(x, x0, gamma, A, c):
    return A * (0.5*gamma)**2 / ((x-x0)**2 + (0.5*gamma)**2) + c

def load_spectrum(fname):
    data = np.loadtxt(fname)
    return data[:, 0], data[:, 1]

def baseline_linear(x, y):
    coef = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
    return np.polyval(coef, x)

def fit_band(x, y, win):
    lo, hi = win
    mask = (x >= lo) & (x <= hi)
    xr, yr = x[mask], y[mask]
    if len(xr) < 10: return None
    idx_max = np.argmax(yr)
    p0 = [float(xr[idx_max]), 40.0, float(yr[idx_max] - np.median(yr)), float(np.median(yr))]
    try:
        popt, _ = curve_fit(lorentz, xr, yr, p0=p0, maxfev=5000)
        x0, gamma, A, c = popt
        return {"pos": float(x0), "fwhm": abs(float(gamma)), "amp": float(A), 
                "c": float(c), "xr": xr, "yr": yr, "fit_y": lorentz(xr, x0, gamma, A, c)}
    except:
        return None

def process_file(filepath):
    filename = os.path.basename(filepath)
    shift, inten = load_spectrum(filepath)
    mask = (shift >= CARBON_RANGE[0]) & (shift <= CARBON_RANGE[1])
    x, y = shift[mask], inten[mask]
    y_corr = y - baseline_linear(x, y)
    
    bands = {"D": (1200, 1500), "G": (1550, 1650), "2D": (2550, 2900)}
    fits = {name: fit_band(x, y_corr, win) for name, win in bands.items()}

    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y_corr, color="gray", alpha=0.5, label="Raw Data")
    for name, f in fits.items():
        if f: plt.plot(f["xr"], f["fit_y"], lw=2, label=f"{name} Fit")
    
    plt.title(f"Raman Analysis: {filename}")
    plt.legend()
    
    output_path = os.path.splitext(filepath)[0] + "_result.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path

def main():
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title="Select Raman TXT files", filetypes=[("Text files", "*.txt")])
    if not files: return
    for f in files:
        try: process_file(f)
        except Exception as e: print(f"Error: {e}")
    messagebox.showinfo("Done", f"Processed {len(files)} files!")

if __name__ == "__main__":
    main()
