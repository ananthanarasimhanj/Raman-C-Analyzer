import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import os

# ========= SETTINGS =========
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
    x0_guess = float(xr[idx_max])
    A_guess = float(yr[idx_max] - np.median(yr))
    gamma_guess = 40.0
    c_guess = float(np.median(yr))
    p0 = [x0_guess, gamma_guess, A_guess, c_guess]

    try:
        popt, _ = curve_fit(lorentz, xr, yr, p0=p0, maxfev=5000)
        x0, gamma, A, c = popt
        fwhm = abs(gamma)

        # Asymmetry calculation from your original code
        y_peak = lorentz(x0, x0, gamma, A, c)
        hm = c + (y_peak - c) / 2
        left, right = xr[xr < x0], xr[xr > x0]
        if len(left) > 0 and len(right) > 0:
            yl, yr_ = lorentz(left, x0, gamma, A, c), lorentz(right, x0, gamma, A, c)
            x_left = left[np.argmin(np.abs(yl - hm))]
            x_right = right[np.argmin(np.abs(yr_ - hm))]
            asym = (x_right - x0) - (x0 - x_left)
        else:
            asym = np.nan

        return {
            "pos": float(x0), "fwhm": float(fwhm), "amp": float(A),
            "c": float(c), "asym": float(asym), "xr": xr, "yr": yr,
            "fit_y": lorentz(xr, x0, gamma, A, c),
        }
    except Exception:
        return None

def classify_from_rules(D, G, TD):
    ID_IG = D["amp"] / G["amp"] if (D and G and G["amp"] != 0) else np.nan
    I2D_IG = TD["amp"] / G["amp"] if (TD and G and G["amp"] != 0) else np.nan
    type_text, notes = "unclassified carbon", []

    if (TD and G and D and TD["fwhm"] < 40 and I2D_IG > 1.5 and ID_IG < 0.1):
        type_text = "high-quality monolayer graphene (strict criteria)"
    else:
        if (TD and G and I2D_IG < 1.0 and TD["fwhm"] > 40 and G["fwhm"] > 30):
            type_text = "defective multilayer / graphite-like sp² carbon"
        if G and G["fwhm"] > 60 and ID_IG > 1.0:
            type_text = "highly disordered / amorphous-like sp² carbon"

    if G:
        if G["pos"] < 1580: notes.append("G band downshifted (disorder/strain/doping).")
        if G["fwhm"] > 30: notes.append("G band significantly broadened.")
    if D:
        notes.append(f"D band present (ID/IG ≈ {ID_IG:.2f})" if not np.isnan(ID_IG) else "D band present.")
    if TD:
        notes.append(f"2D broad (FWHM ≈ {TD['fwhm']:.1f})" if not np.isnan(I2D_IG) else "2D broad.")

    return type_text, ID_IG, I2D_IG, notes

class RamanApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Raman Pro Analyzer")
        self.root.geometry("600x450")
        
        tk.Label(root, text="Raman Analysis Tool", font=("Arial", 16, "bold")).pack(pady=10)
        self.btn = tk.Button(root, text="Select Files & Process", command=self.start, bg="#2196F3", fg="white", font=("Arial", 11, "bold"), padx=20, pady=10)
        self.btn.pack(pady=10)
        
        tk.Label(root, text="Activity Log:").pack(anchor="w", padx=25)
        self.log = scrolledtext.ScrolledText(root, width=65, height=15)
        self.log.pack(padx=20, pady=5)

    def log_m(self, m):
        self.log.insert(tk.END, m + "\n"); self.log.see(tk.END); self.root.update_idletasks()

    def run_analysis(self, filepath):
        fname = os.path.basename(filepath)
        data = np.loadtxt(filepath)
        shift, inten = data[:, 0], data[:, 1]
        mask = (shift >= CARBON_RANGE[0]) & (shift <= CARBON_RANGE[1])
        x, y = shift[mask], inten[mask]
        
        y_corr = y - np.polyval(np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1), x)
        bands = {"D": (1200, 1500), "G": (1550, 1650), "2D": (2550, 2900)}
        fits = {name: fit_band(x, y_corr, win) for name, win in bands.items()}
        
        carbon_type, ID_IG, I2D_IG, notes = classify_from_rules(fits["D"], fits["G"], fits["2D"])

        # --- REPRODUCING YOUR ORIGINAL PLOT LAYOUT ---
        fig = plt.figure(figsize=(10, 5))
        ax_spec = fig.add_axes([0.08, 0.12, 0.60, 0.8])
        ax_spec.plot(x, y_corr, color="k", lw=1.2)
        
        colors = {"D": "tab:blue", "G": "tab:green", "2D": "tab:red"}
        for name, f in fits.items():
            if f:
                ax_spec.plot(f["xr"], f["fit_y"], color=colors[name], lw=1.5)
                ax_spec.axvline(f["pos"], color=colors[name], ls="--", alpha=0.6)

        ax_spec.set_xlabel("Raman shift (cm$^{-1}$)")
        ax_spec.set_ylabel("Intensity (a.u.)")
        ax_spec.set_title(f"Raman spectrum: {fname}")

        # Summary text panel (exactly as you had it)
        ax_text = fig.add_axes([0.72, 0.12, 0.26, 0.8]); ax_text.axis("off")
        lines = [f"File: {fname}", f"Laser: {LASER_NM} nm", "", "Band fits:"]
        for name in ["D", "G", "2D"]:
            f = fits.get(name)
            lines.append(f"  {name}: {f['pos']:.1f} cm⁻¹, FWHM {f['fwhm']:.1f}" if f else f"  {name}: not fitted")
        
        lines.extend(["", f"Type: {carbon_type}"])
        if not np.isnan(ID_IG): lines.append(f"ID/IG ≈ {ID_IG:.2f} (height)")
        if not np.isnan(I2D_IG): lines.append(f"I2D/IG ≈ {I2D_IG:.2f} (height)")
        if notes:
            lines.extend(["", "Notes:"])
            for n in notes: lines.append(f"- {n}")

        ax_text.text(0.0, 1.0, "\n".join(lines), ha="left", va="top", fontsize=8)
        
        out_path = os.path.splitext(filepath)[0] + "_result.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        return out_path

    def start(self):
        files = filedialog.askopenfilenames(title="Select Raman TXT files", filetypes=[("Text files", "*.txt")])
        if not files: return
        self.log_m(f"--- Starting Batch ({len(files)} files) ---")
        for f in files:
            try:
                path = self.run_analysis(f)
                self.log_m(f"SAVED: {os.path.basename(path)}")
            except Exception as e:
                self.log_m(f"FAILED {os.path.basename(f)}: {e}")
        self.log_m("--- Finished ---")
        messagebox.showinfo("Complete", "Analysis images saved to data folders.")

if __name__ == "__main__":
    root = tk.Tk(); app = RamanApp(root); root.mainloop()
