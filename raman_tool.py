import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from fpdf import FPDF
import os
import subprocess

# ========= SETTINGS =========
VERSION = os.getenv("APP_VERSION", "1.3.0") 
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
        # Asymmetry for 3D stacking check
        left, right = xr[xr < x0], xr[xr > x0]
        asym = 0
        if len(left) > 0 and len(right) > 0:
            hm = c + (A / 2)
            xl = left[np.argmin(np.abs(lorentz(left, x0, gamma, A, c) - hm))]
            xr_ = right[np.argmin(np.abs(lorentz(right, x0, gamma, A, c) - hm))]
            asym = (xr_ - x0) - (x0 - xl)
        return {"pos": x0, "fwhm": abs(gamma), "amp": A, "c": c, "asym": asym, "xr": xr, "fit_y": y_fit}
    except: return None

def get_audit_logic(D, G, TD, ID_IG, I2D_IG):
    steps = []
    # 1. Monolayer
    m_pass = (TD and I2D_IG > 1.5 and TD["fwhm"] < 40)
    steps.append({"name": "Monolayer Graphene", "status": "PASSED" if m_pass else "FAILED", 
                  "reason": f"I2D/IG={I2D_IG:.2f} (Need >1.5), FWHM={TD['fwhm'] if TD else 0:.1f}"})
    # 2. Graphite
    g_pass = (not m_pass and TD and I2D_IG < 0.75 and TD["asym"] > 4)
    steps.append({"name": "Pristine Graphite", "status": "PASSED" if g_pass else "FAILED", 
                  "reason": f"Asymmetry={TD['asym'] if TD else 0:.1f} (Need >4)"})
    # 3. Multilayer
    ml_pass = (not m_pass and not g_pass and TD and I2D_IG >= 0.5)
    steps.append({"name": "Multilayer Graphene", "status": "PASSED" if ml_pass else "FAILED", 
                  "reason": f"I2D/IG={I2D_IG:.2f} (Falls in 0.5-1.5 range)"})
    # 4. Turbostratic
    t_pass = (not m_pass and not g_pass and not ml_pass and TD and I2D_IG < 0.5)
    steps.append({"name": "Turbostratic Carbon", "status": "PASSED" if t_pass else "FAILED", 
                  "reason": "Symmetric 2D with low I2D/IG"})
    
    return steps

def classify_carbon(D, G, TD):
    ID_IG = D["amp"] / G["amp"] if (D and G and G["amp"] != 0) else np.nan
    I2D_IG = TD["amp"] / G["amp"] if (TD and G and G["amp"] != 0) else np.nan
    La = (2.4e-10) * (LASER_NM**4) * (1.0 / ID_IG) if (not np.isnan(ID_IG) and ID_IG > 0) else np.nan
    
    audit = get_audit_logic(D, G, TD, ID_IG, I2D_IG)
    
    # Final Decision based on Audit
    ctype = "Unclassified"
    for step in audit:
        if step["status"] == "PASSED":
            ctype = step["name"]
            break
            
    if ID_IG > 0.15 and "Pristine" not in ctype:
        ctype = "Defective " + ctype

    return ctype, ID_IG, I2D_IG, La, audit

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, f'Raman Analysis Audit Report - v{VERSION}', 0, 1, 'C')
        self.ln(5)

def create_pdf(fname, ctype, idig, i2dig, La, fits, audit, img_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 10, f"Sample Name: {fname}", ln=True)
    pdf.cell(0, 10, f"Final Conclusion: {ctype}", ln=True)
    pdf.ln(5)

    # Audit Table
    pdf.set_fill_color(230, 230, 230)
    pdf.cell(50, 8, "Verification Step", 1, 0, 'C', True)
    pdf.cell(30, 8, "Status", 1, 0, 'C', True)
    pdf.cell(110, 8, "Reasoning", 1, 1, 'C', True)
    
    pdf.set_font("Arial", size=9)
    for step in audit:
        pdf.cell(50, 8, step["name"], 1)
        pdf.set_text_color(0, 128, 0) if step["status"] == "PASSED" else pdf.set_text_color(200, 0, 0)
        pdf.cell(30, 8, step["status"], 1, 0, 'C')
        pdf.set_text_color(0, 0, 0)
        pdf.cell(110, 8, step["reason"], 1, 1)

    pdf.ln(10)
    pdf.image(img_path, x=10, w=190)
    
    p_path = img_path.replace("_plot.png", "_REPORT.pdf")
    pdf.output(p_path)
    return p_path

class RamanApp:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Raman Pro v{VERSION}")
        self.root.geometry("600x480")
        tk.Label(root, text=f"Raman Pro Analyzer v{VERSION}", font=("Arial", 16, "bold")).pack(pady=10)
        tk.Button(root, text="Process Files (Batch)", command=self.start, bg="#2196F3", fg="white", font=("Arial", 10, "bold"), padx=20, pady=10).pack(pady=10)
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

        fig = plt.figure(figsize=(12, 6))
        ax_spec = fig.add_axes([0.08, 0.12, 0.52, 0.78])
        ax_spec.plot(x, y_corr, color="gray", alpha=0.3)
        for n, f in fits.items():
            if f: ax_spec.plot(f["xr"], f["fit_y"], lw=2, label=f"{n} Fit")
        ax_spec.set_title(f"Sample: {fname}")
        ax_spec.legend()

        ax_text = fig.add_axes([0.65, 0.12, 0.32, 0.78]); ax_text.axis("off")
        summary = [f"File: {fname}", f"Ver: {VERSION}", "", "PEAK DATA (Pos/FWHM):"]
        for n in ["D", "G", "2D"]:
            f = fits.get(n)
            summary.append(f"  {n}: {f['pos']:.1f}/{f['fwhm']:.1f}" if f else f"  {n}: N/A")
        summary.extend(["", f"ID/IG: {idig:.2f}", f"I2D/IG: {i2dig:.2f}", "", f"Conclusion: {ctype}"])
        ax_text.text(0, 1, "\n".join(summary), va="top", fontsize=9)
        
        base = os.path.splitext(filepath)[0]
        img_p = f"{base}_v{VERSION}_plot.png"
        plt.savefig(img_p, dpi=300); plt.close()
        create_pdf(fname, ctype, idig, i2dig, La, fits, audit, img_p)
        return img_p

    def start(self):
        files = filedialog.askopenfilenames()
        if not files: return
        res_dir = os.path.dirname(files[0])
        for f in files:
            try: p = self.run_analysis(f); self.log_m(f"DONE: {os.path.basename(p)}")
            except Exception as e: self.log_m(f"ERROR: {e}")
        if messagebox.askyesno("Finished", "Open results folder?"):
            os.startfile(res_dir) if os.name == 'nt' else subprocess.call(['open', res_dir])

if __name__ == "__main__":
    root = tk.Tk(); app = RamanApp(root); root.mainloop()
