import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os
import subprocess

# ========= SETTINGS =========
VERSION = os.getenv("APP_VERSION", "1.2.0") 
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
        
        # Calculate Asymmetry for Graphite Verification
        y_fit = lorentz(xr, x0, gamma, A, c)
        half_max = c + (A / 2)
        left_side = xr[xr < x0]
        right_side = xr[xr > x0]
        asym = 0
        if len(left_side) > 0 and len(right_side) > 0:
            x_l = left_side[np.argmin(np.abs(lorentz(left_side, x0, gamma, A, c) - half_max))]
            x_r = right_side[np.argmin(np.abs(lorentz(right_side, x0, gamma, A, c) - half_max))]
            asym = (x_r - x0) - (x0 - x_l)
            
        return {"pos": float(x0), "fwhm": float(abs(gamma)), "amp": float(A), "c": float(c), 
                "asym": asym, "xr": xr, "yr": yr, "fit_y": y_fit}
    except:
        return None

def classify_carbon(D, G, TD):
    ID_IG = D["amp"] / G["amp"] if (D and G and G["amp"] != 0) else np.nan
    I2D_IG = TD["amp"] / G["amp"] if (TD and G and G["amp"] != 0) else np.nan
    La = (2.4e-10) * (LASER_NM**4) * (1.0 / ID_IG) if (not np.isnan(ID_IG) and ID_IG > 0) else np.nan
    
    ctype = "Unclassified Carbon"
    evidence = []

    # 1. Monolayer Graphene
    if TD and I2D_IG > 1.5 and TD["fwhm"] < 40:
        ctype = "Monolayer Graphene"; evidence = ["High I2D/IG ratio", "Narrow symmetric 2D"]
    
    # 2. Pristine Graphite
    elif TD and I2D_IG < 0.7 and TD["asym"] > 4 and (not D or ID_IG < 0.15):
        ctype = "Pristine Graphite"; evidence = ["Asymmetric 2D (3D-stacking)", "Low defect density"]

    # 3. Multilayer Graphene
    elif TD and 0.7 <= I2D_IG <= 1.3:
        ctype = "Multilayer Graphene"; evidence = ["Intermediate I2D/IG", "Broadened 2D"]

    # 4. Turbostratic Carbon
    elif TD and I2D_IG < 0.5 and TD["fwhm"] > 60:
        ctype = "Turbostratic Carbon"; evidence = ["Broad symmetric 2D", "Low stacking order"]

    # 5. Carbon Black / Soot
    elif G and G["fwhm"] > 60 and ID_IG > 1.0:
        ctype = "Carbon Black / Soot"; evidence = ["High ID/IG ratio", "G-band broadening"]

    # 6. Amorphous Carbon
    if G and G["fwhm"] > 85:
        ctype = "Amorphous Carbon"; evidence = ["Extreme G-FWHM", "High structural disorder"]

    # Defect Modifier
    if not np.isnan(ID_IG) and ID_IG > 0.2 and "Pristine" not in ctype and "Amorphous" not in ctype:
        ctype = f"Defective {ctype}"

    return ctype, ID_IG, I2D_IG, La, evidence

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
        ctype, idig, i2dig, La, evidence = classify_carbon(fits["D"], fits["G"], fits["2D"])

        fig = plt.figure(figsize=(12, 6))
        ax_spec = fig.add_axes([0.08, 0.12, 0.52, 0.78])
        ax_spec.plot(x, y_corr, color="gray", alpha=0.3, label="Raw Data")
        
        colors = {"D": "tab:blue", "G": "tab:green", "2D": "tab:red"}
        for n, f in fits.items():
            if f: ax_spec.plot(f["xr"], f["fit_y"], color=colors[n], lw=2, label=f"{n} Peak")
        
        ax_spec.set_xlabel("Raman shift (cm⁻¹)"); ax_spec.set_ylabel("Intensity (a.u.)")
        ax_spec.set_title(f"Sample: {fname}")
        ax_spec.legend()

        ax_text = fig.add_axes([0.65, 0.12, 0.32, 0.78]); ax_text.axis("off")
        summary = [f"File: {fname}", f"Ver: {VERSION}", f"Laser: {LASER_NM} nm", "", "BAND FITS (Pos / FWHM):"]
        for n in ["D", "G", "2D"]:
            f = fits.get(n)
            summary.append(f"  {n}: {f['pos']:.1f} / {f['fwhm']:.1f} cm⁻¹" if f else f"  {n}: N/A")
        
        summary.extend(["", "RATIOS:", f"  ID/IG: {idig:.2f}" if not np.isnan(idig) else "  ID/IG: N/A",
                        f"  I2D/IG: {i2dig:.2f}" if not np.isnan(i2dig) else "  I2D/IG: N/A", "",
                        f"Crystallite Size (La):", f"  {La:.1f} nm" if not np.isnan(La) else "  N/A", "",
                        "CONCLUSION:", f"  {ctype}", ""])
        
        if evidence:
            summary.append("Verified By:")
            for e in evidence: summary.append(f"  - {e}")

        ax_text.text(0, 1, "\n".join(summary), va="top", fontsize=9, family='monospace')
        
        out = os.path.splitext(filepath)[0] + f"_v{VERSION}_result.png"
        plt.savefig(out, dpi=300); plt.close()
        return out

    def start(self):
        files = filedialog.askopenfilenames(title="Select Batch Files from One Folder")
        if not files: return
        res_dir = os.path.dirname(files[0])
        self.log_m(f"--- Processing v{VERSION} ---")
        for f in files:
            try: p = self.run_analysis(f); self.log_m(f"SUCCESS: {os.path.basename(p)}")
            except Exception as e: self.log_m(f"FAILED: {e}")
        if messagebox.askyesno("Complete", "Analysis finished. Open results folder?"):
            os.startfile(res_dir) if os.name == 'nt' else subprocess.call(['open', res_dir])

if __name__ == "__main__":
    root = tk.Tk(); app = RamanApp(root); root.mainloop()
