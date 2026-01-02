import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import subprocess

# ========= SETTINGS =========
VERSION = os.getenv("APP_VERSION", "1.1.0") 
LASER_NM = 532
CARBON_RANGE = (1000, 3200)
# ============================

def lorentz(x, x0, gamma, A, c):
    return A * (0.5*gamma)**2 / ((x-x0)**2 + (0.5*gamma)**2) + c

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
        return {"pos": float(x0), "fwhm": float(abs(gamma)), "amp": float(A), "c": float(c), "xr": xr, "yr": yr, "fit_y": lorentz(xr, x0, gamma, A, c)}
    except:
        return None

def classify_carbon(D, G, TD):
    ID_IG = D["amp"] / G["amp"] if (D and G and G["amp"] != 0) else np.nan
    I2D_IG = TD["amp"] / G["amp"] if (TD and G and G["amp"] != 0) else np.nan
    type_text, notes = "Unclassified Carbon", []
    La = np.nan

    if not np.isnan(ID_IG) and ID_IG > 0:
        La = (2.4e-10) * (LASER_NM**4) * (1.0 / ID_IG)

    if (TD and G and D and TD["fwhm"] < 40 and I2D_IG > 1.5 and ID_IG < 0.1):
        type_text = "High-Quality Monolayer Graphene"
    elif TD and G:
        if I2D_IG < 0.5 and TD["fwhm"] > 60:
            type_text = "Bulk / 3D Graphite"
        elif I2D_IG < 1.0 and TD["fwhm"] > 45:
            type_text = "Turbostratic / Multilayer Carbon"
        elif I2D_IG > 1.0 and TD["fwhm"] > 40:
            type_text = "Few-Layer Graphene"

    if G and G["fwhm"] > 60 and ID_IG > 1.0:
        type_text = "Amorphous / Highly Disordered Carbon"

    if not np.isnan(La): notes.append(f"Crystallite Size (La) ≈ {La:.1f} nm")
    if G and G["pos"] < 1580: notes.append("G-band red-shifted (strain/doping).")
    
    return type_text, ID_IG, I2D_IG, La, notes

class RamanApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Raman Pro Analyzer v{VERSION}")
        self.root.geometry("600x480")
        tk.Label(root, text=f"Raman Analysis Tool v{VERSION}", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Button(root, text="Select Files & Process", command=self.start, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), padx=20, pady=10).pack(pady=10)
        self.log = scrolledtext.ScrolledText(root, width=65, height=15)
        self.log.pack(padx=20, pady=5)

    def log_m(self, m):
        self.log.insert(tk.END, m + "\n"); self.log.see(tk.END); self.root.update_idletasks()

    def run_analysis(self, filepath):
        data = np.loadtxt(filepath)
        shift, inten = data[:, 0], data[:, 1]
        mask = (shift >= CARBON_RANGE[0]) & (shift <= CARBON_RANGE[1])
        x, y = shift[mask], inten[mask]
        y_corr = y - np.polyval(np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1), x)
        fits = {n: fit_band(x, y_corr, w) for n, w in {"D": (1200, 1500), "G": (1550, 1650), "2D": (2550, 2900)}.items()}
        
        ctype, idig, i2dig, La, notes = classify_carbon(fits["D"], fits["G"], fits["2D"])

        fig = plt.figure(figsize=(11, 5))
        ax_spec = fig.add_axes([0.08, 0.12, 0.55, 0.8])
        ax_spec.plot(x, y_corr, color="k", lw=1.2)
        for n, f in fits.items():
            if f: ax_spec.plot(f["xr"], f["fit_y"], lw=1.5, label=n)
        ax_spec.set_title(os.path.basename(filepath))
        ax_spec.legend()

        ax_text = fig.add_axes([0.68, 0.12, 0.30, 0.8]); ax_text.axis("off")
        summary = [f"Ver: {VERSION}", f"Laser: {LASER_NM}nm", "", "Parameters:"]
        for n, f in fits.items(): 
            summary.append(f" {n}: {f['pos']:.1f} cm⁻¹" if f else f" {n}: N/A")
        summary.extend(["", f"ID/IG: {idig:.2f}" if not np.isnan(idig) else "ID/IG: N/A"])
        summary.extend(["", f"Conclusion:", ctype, ""])
        summary.extend(notes)
        ax_text.text(0, 1, "\n".join(summary), va="top", fontsize=8)
        
        out = os.path.splitext(filepath)[0] + "_v110_result.png"
        plt.savefig(out, dpi=300); plt.close()
        return out

    def start(self):
        files = filedialog.askopenfilenames(title="Select Files from a Single Folder")
        if not files: return
        
        # Determine the folder of the files
        result_dir = os.path.dirname(files[0])
        
        self.log_m(f"--- Processing Batch v{VERSION} ---")
        for f in files:
            try: 
                path = self.run_analysis(f)
                self.log_m(f"SAVED: {os.path.basename(path)}")
            except Exception as e: 
                self.log_m(f"ERROR: {e}")
        
        self.log_m("--- Finished ---")
        
        # Ask to open folder
        if messagebox.askyesno("Batch Complete", f"Processed {len(files)} files.\n\nWould you like to open the result folder?"):
            # Cross-platform way to open folder
            if os.name == 'nt': # Windows
                os.startfile(result_dir)
            elif os.name == 'posix': # Mac/Linux
                subprocess.call(['open', result_dir])

if __name__ == "__main__":
    root = tk.Tk(); app = RamanApp(root); root.mainloop()
