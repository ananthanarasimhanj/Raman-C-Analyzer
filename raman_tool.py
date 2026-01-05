import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import subprocess

# ========= SETTINGS =========
VERSION = os.getenv("APP_VERSION", "1.2.1") 
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
        y_fit = lorentz(xr, x0, gamma, A, c)
        left, right = xr[xr < x0], xr[xr > x0]
        asym = 0
        if len(left) > 0 and len(right) > 0:
            hm = c + (A / 2)
            xl = left[np.argmin(np.abs(lorentz(left, x0, gamma, A, c) - hm))]
            xr_ = right[np.argmin(np.abs(lorentz(right, x0, gamma, A, c) - hm))]
            asym = (xr_ - x0) - (x0 - xl)
        return {"pos": x0, "fwhm": abs(gamma), "amp": A, "c": c, "asym": asym, "xr": xr, "fit_y": y_fit}
    except: return None

def generate_audit(D, G, TD, ID_IG, I2D_IG):
    audit = []
    # 1. Monolayer Check
    if I2D_IG > 1.5: audit.append("[✓] I2D/IG > 1.5")
    else: audit.append(f"[x] Not Monolayer: I2D/IG is {I2D_IG:.2f}")
    
    # 2. Graphite Check
    if TD and TD["asym"] > 4: audit.append("[✓] 2D Asymmetric (3D)")
    else: audit.append("[x] Not 3D Graphite: 2D symmetric")
    
    # 3. Defect Check
    if ID_IG > 0.2: audit.append(f"[!] Defective: ID/IG is {ID_IG:.2f}")
    else: audit.append("[✓] Low defects")
    
    return audit

def classify_carbon(D, G, TD):
    ID_IG = D["amp"] / G["amp"] if (D and G and G["amp"] != 0) else np.nan
    I2D_IG = TD["amp"] / G["amp"] if (TD and G and G["amp"] != 0) else np.nan
    La = (2.4e-10) * (LASER_NM**4) * (1.0 / ID_IG) if (not np.isnan(ID_IG) and ID_IG > 0) else np.nan
    
    notes = []
    # Hierarchy Logic
    if TD and I2D_IG > 1.5 and TD["fwhm"] < 40:
        ctype = "Monolayer Graphene"
    elif TD and I2D_IG < 0.7 and TD["asym"] > 4:
        ctype = "Pristine Graphite"
    elif TD and I2D_IG >= 0.5:
        ctype = "Multilayer Graphene"
    elif TD and I2D_IG < 0.5:
        ctype = "Turbostratic Carbon"
    elif G and G["fwhm"] > 60:
        ctype = "Soot / Carbon Black"
    else:
        ctype = "Disordered Carbon"

    if ID_IG > 0.15 and "Pristine" not in ctype:
        ctype = "Defective " + ctype

    audit = generate_audit(D, G, TD, ID_IG, I2D_IG)
    return ctype, ID_IG, I2D_IG, La, audit

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
        fname = os.path.basename(filepath)
        data = np.loadtxt(filepath)
        x_raw, y_raw = data[:, 0], data[:, 1]
        mask = (x_raw >= CARBON_RANGE[0]) & (x_raw <= CARBON_RANGE[1])
        x, y = x_raw[mask], y_raw[mask]
        y_corr = y - np.polyval(np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1), x)
        
        fits = {n: fit_band(x, y_corr, w) for n, w in {"D": (1200, 1500), "G": (1550, 1650), "2D": (2550, 2900)}.items()}
        ctype, idig, i2dig, La, audit = classify_carbon(fits["D"], fits["G"], fits["2D"])

        # Plotting
        fig = plt.figure(figsize=(12, 6))
        ax_spec = fig.add_axes([0.08, 0.12, 0.52, 0.78])
        ax_spec.plot(x, y_corr, color="gray", alpha=0.3)
        colors = {"D": "tab:blue", "G": "tab:green", "2D": "tab:red"}
        for n, f in fits.items():
            if f: ax_spec.plot(f["xr"], f["fit_y"], color=colors[n], lw=2, label=n)
        ax_spec.set_title(f"Sample: {fname}")
        ax_spec.legend()

        ax_text = fig.add_axes([0.65, 0.12, 0.32, 0.78]); ax_text.axis("off")
        summary = [f"File: {fname}", f"Analyzer: v{VERSION}", "", "FITS (Pos/FWHM):"]
        for n in ["D", "G", "2D"]:
            f = fits.get(n)
            summary.append(f"  {n}: {f['pos']:.1f}/{f['fwhm']:.1f}" if f else f"  {n}: N/A")
        
        summary.extend(["", "RATIOS:", f"  ID/IG: {idig:.2f}", f"  I2D/IG: {i2dig:.2f}", "", f"Conclusion: {ctype}", "", "AUDIT LOG:"])
        summary.extend(audit)
        ax_text.text(0, 1, "\n".join(summary), va="top", fontsize=8, family='monospace')
        
        out_img = os.path.splitext(filepath)[0] + "_result.png"
        plt.savefig(out_img, dpi=300); plt.close()

        # Save Text Report
        with open(os.path.splitext(filepath)[0] + "_REPORT.txt", "w") as f:
            f.write(f"Raman Analysis Report - v{VERSION}\n" + "="*30 + "\n")
            f.write("\n".join(summary))
            
        return out_img

    def start(self):
        files = filedialog.askopenfilenames()
        if not files: return
        res_dir = os.path.dirname(files[0])
        for f in files:
            try: p = self.run_analysis(f); self.log_m(f"DONE: {os.path.basename(p)}")
            except Exception as e: self.log_m(f"ERROR: {e}")
        if messagebox.askyesno("Complete", "Open results?"):
            os.startfile(res_dir) if os.name == 'nt' else subprocess.call(['open', res_dir])

if __name__ == "__main__":
    root = tk.Tk(); app = RamanApp(root); root.mainloop()
