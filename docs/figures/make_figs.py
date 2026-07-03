"""Generate the illustrative figures for the pyMack LST document set
(mean_boundary_layer.tex, disturbance_equations.tex, numerical_methods.tex).

All figures are written as vector PDFs into this directory. Data-driven panels
(base flow, Chebyshev grid, differentiation matrix, operator occupancy, neutral
field) use the real pyMack code / digitised grids so the document shows the
actual objects, not cartoons. Schematic panels (setup, ansatz, EVP planes,
BC rows, workflow) are drawn to scale and clearly labelled.

Font rule (repo standard): labels >= 14, ticks >= 12, titles >= 16.
Run:  PYMACK_NO_BANNER=1 python docs/figures/make_figs.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130,
    "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
    "axes.titlesize": 16, "legend.fontsize": 12.5, "font.family": "DejaVu Sans",
    "axes.linewidth": 1.0, "lines.linewidth": 2.2,
})
INK = "#111111"; PM = "#1f8f6b"; REF = "#d55e00"; HOT = "#c0392b"; COOL = "#2c6fbb"
FILL = "#dcebe4"; GREY = "#888888"

def save(fig, name):
    fig.savefig(os.path.join(HERE, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# Worked example: a clean, domain-stationary, wall-trapped Mack (2nd) mode.
WE = dict(Ma=6.0, Re=5500.0, al=0.174, c_disc=0.9301 + 0.0200j)
_WE_CACHE = {}
def we_solve(ymf=10.0, N=180):
    key = (ymf, N)
    if key not in _WE_CACHE:
        from pymack import make_flatplate_profile
        from pymack.temporal_solver import solve_temporal_2d
        from pymack.scales import delta_star_over_lstar
        p = make_flatplate_profile(WE["Ma"]); d = delta_star_over_lstar(p)
        ev, vec, y = solve_temporal_2d(p, WE["al"], WE["Re"], WE["Ma"],
                                             N=N, y_max=ymf * d, length_scale="L_star")
        _WE_CACHE[key] = (ev, vec, y, p, d)
    return _WE_CACHE[key]
def we_fields(ev, vec, y, idx):
    """Return (|u|,|v|,|T|,|p|) of mode idx, each normalised, and the grid (wall=0..free)."""
    n = len(y)
    f = [np.abs(vec[k * n:(k + 1) * n, idx]) for k in range(4)]
    return [fi / (fi.max() + 1e-30) for fi in f], n
def we_disc_index(ev):
    return int(np.argmin(np.abs(ev - WE["c_disc"])))


# ----------------------------------------------------------------------
def fig_setup():
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.linspace(0, 10, 400)
    delta = 1.35 * np.sqrt(np.maximum(x, 1e-6))
    ax.plot(x, delta, color=COOL, lw=2.4)
    ax.fill_between(x, 0, delta, color=FILL, alpha=0.6, zorder=0)
    ax.plot([0, 10], [0, 0], color=INK, lw=3.0)            # plate
    ax.annotate("boundary-layer edge", xy=(8.2, 1.35 * np.sqrt(8.2)), xytext=(4.4, 3.35),
                color=COOL, fontsize=13,
                arrowprops=dict(arrowstyle="-|>", color=COOL, lw=1.2))
    # velocity profile at a station
    xs = 6.0
    d = 1.35 * np.sqrt(xs)
    yy = np.linspace(0, d, 9)
    uu = np.tanh(2.4 * yy / d)
    for yi, ui in zip(yy, uu):
        ax.add_patch(FancyArrowPatch((xs, yi), (xs + 2.0 * ui, yi),
                     arrowstyle="-|>", mutation_scale=11, color=HOT, lw=1.6))
    ax.plot(xs + 2.0 * uu, yy, color=HOT, lw=2.0)
    ax.text(xs + 0.1, d + 0.15, r"$\overline{U}(y)$", color=HOT, fontsize=15)
    # instability wave riding in the layer, AMPLIFYING downstream (~ e^{+sigma x})
    xw = np.linspace(1.0, 9.6, 600)
    env = 0.045 * np.exp(0.30 * (xw - 1.0))
    yw = 0.78 + env * np.sin(2 * np.pi * xw / 1.5)
    ax.plot(xw, yw, color=PM, lw=2.0)
    ax.plot(xw, 0.78 + env, color=PM, lw=1.0, ls="--", alpha=0.7)
    ax.plot(xw, 0.78 - env, color=PM, lw=1.0, ls="--", alpha=0.7)
    ax.text(0.9, 1.35, r"disturbance $q'$ amplifying $\sim e^{+\sigma x}$", color=PM, fontsize=12.5)
    # edge velocity arrow
    ax.add_patch(FancyArrowPatch((0.3, 4.05), (2.3, 4.05), arrowstyle="-|>",
                 mutation_scale=14, color=INK, lw=1.8))
    ax.text(0.3, 4.2, r"$U_e,\,T_e,\,\rho_e$ (edge)", fontsize=13)
    ax.annotate("", xy=(xs - 0.55, d), xytext=(xs - 0.55, 0),
                arrowprops=dict(arrowstyle="<->", color=GREY, lw=1.4))
    ax.text(xs - 1.5, d / 2, r"$\delta(x)$", color=GREY, fontsize=14)
    ax.set_xlim(-0.2, 10.6); ax.set_ylim(-0.25, 4.5)
    ax.set_xlabel(r"streamwise coordinate  $x$")
    ax.set_ylabel(r"wall-normal  $y$")
    ax.set_title("Physical setup: a wave on a compressible boundary layer")
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig_setup.pdf")


def fig_ansatz():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    y = np.linspace(0, 6, 400)
    q = (y) * np.exp(-1.1 * y); q /= q.max()
    a1.plot(q, y, color=PM, lw=2.6)
    a1.fill_betweenx(y, 0, q, color=FILL, alpha=0.7)
    a1.set_xlabel(r"$|\hat q(y)|$  (amplitude)")
    a1.set_ylabel(r"wall-normal  $y$")
    a1.set_title(r"Eigenfunction  $\hat q(y)$")
    a1.text(0.36, 4.6, "decays to 0\nat freestream", fontsize=12, color=GREY)
    a1.text(0.02, 0.18, "= 0 at wall", fontsize=12, color=GREY)
    a1.grid(alpha=0.25)
    x = np.linspace(0, 4 * np.pi, 500)
    a2.plot(x, np.sin(x), color=INK, lw=2.4)
    a2.axhline(0, color=GREY, lw=0.8)
    a2.annotate("", xy=(2 * np.pi, 1.28), xytext=(0, 1.28),
                arrowprops=dict(arrowstyle="<->", color=REF, lw=1.6))
    a2.text(np.pi * 0.7, 1.36, r"wavelength $\lambda=2\pi/\alpha$", color=REF, fontsize=12.5)
    a2.add_patch(FancyArrowPatch((3.2, -1.35), (4.7, -1.35), arrowstyle="-|>",
                 mutation_scale=13, color=PM, lw=1.8))
    a2.text(3.0, -1.2, r"phase speed $c_r=\omega/\alpha$", color=PM, fontsize=12.5)
    a2.set_ylim(-1.7, 1.7)
    a2.set_xlabel(r"streamwise coordinate  $x$")
    a2.set_title(r"Travelling wave  $e^{\,i(\alpha x-\omega t)}$")
    a2.set_yticks([-1, 0, 1])
    fig.tight_layout()
    save(fig, "fig_ansatz.pdf")


def fig_baseflow():
    from pymack import make_flatplate_profile
    # Representative hypersonic cooled-wall case: M=6, T_e=54 K, isothermal wall
    # at T_w=300 K (T_w/T_e=5.56, below the adiabatic recovery ~7.1 T_e).
    p = make_flatplate_profile(6.0, T_edge=54.0, T_wall=300.0)
    eta_max = 2.5                      # edge is at eta ~ 1.5; leave a freestream margin
    eta = np.linspace(0, eta_max, 400)
    bf = p(eta)
    U = np.asarray(bf["U"]); T = np.asarray(bf["T"]); dU = np.asarray(bf["dU"])
    g = dU / T                         # rho-bar * dU/deta  (rho-bar = 1/T-bar)
    dg = np.gradient(g, eta)
    etas = None                        # OUTERMOST generalised inflection point below the
    for i in range(4, len(eta) - 1):   # edge (cooled walls add a near-wall crossing at the
        if dg[i - 1] * dg[i] < 0 and 0.3 < U[i] < 0.999:   # interior T max; skip it, and
            etas = eta[i]              # skip freestream noise where U=1 and dU~0)

    # Single panel: mean profiles in the similarity (generalised) coordinate.
    # Conversion to physical y is Eq. (ytransform)/(ydim) in mean_boundary_layer.tex.
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.plot(U, eta, color=COOL, lw=2.6, label=r"$\overline{U}/U_e$")
    ax.plot(T, eta, color=HOT, lw=2.6, label=r"$\overline{T}/T_e$")
    if etas is not None:
        ax.axhline(etas, color="#555", ls=":", lw=1.4, zorder=1)
        ax.scatter([np.interp(etas, eta, U)], [etas], s=45, color="#222", zorder=6)
        ax.text(T.max() * 0.50, etas + 0.20,
                r"$(\overline{\rho}\,\overline{U}\,\!\!^\prime)^\prime=0$",
                fontsize=13, color="#222")
    ax.set_xlabel(r"$\overline{U}/U_e,\ \ \overline{T}/T_e$")
    ax.set_ylabel(r"similarity coordinate  $\eta$")
    ax.set_xlim(0, max(1.05, T.max() * 1.06)); ax.set_ylim(0, eta_max)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_baseflow.pdf")


def fig_cheb_grid():
    from pymack.spectral import chebyshev_points, map_domain
    N = 24; ymax = 6.0; L = 1.0
    xi = chebyshev_points(N)
    y, _, _ = map_domain(xi, ymax, L)
    fig, ax = plt.subplots(figsize=(9.0, 3.7))
    # computational line at top, physical line at bottom
    yc, yp = 1.0, 0.0
    ax.plot([-1, 1], [yc, yc], color=GREY, lw=1.2)
    # map physical y (0..ymax) onto the same horizontal extent [-1,1] for display
    xp = -1 + 2 * (y / ymax)
    ax.plot([-1, 1], [yp, yp], color=GREY, lw=1.2)
    for xii, xpi in zip(xi, xp):
        ax.plot([xii, xpi], [yc, yp], color="#cfd8d3", lw=0.7, zorder=0)
    ax.scatter(xi, np.full_like(xi, yc), s=42, color=COOL, zorder=3)
    ax.scatter(xp, np.full_like(xp, yp), s=42, color=PM, zorder=3)
    ax.text(-1.02, yc + 0.16, r"computational  $\xi_j=\cos(\pi j/n)\in[-1,1]$",
            fontsize=12.5, color=COOL)
    ax.text(-1.02, yp - 0.28, r"physical  $y\in[0,y_{\max}]$ (clustered at wall)",
            fontsize=12.5, color=PM)
    ax.annotate("wall  $y=0$", xy=(-1, yp), xytext=(-1.0, -0.62), fontsize=12,
                ha="center", arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.annotate(r"freestream  $y=y_{\max}$", xy=(1, yp), xytext=(0.95, -0.62),
                fontsize=12, ha="center", arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.8, 1.5)
    ax.axis("off")
    save(fig, "fig_cheb_grid.pdf")


def fig_diffmat():
    from pymack.spectral import chebyshev_D, physical_derivatives
    N = 24
    y, D1, D2 = physical_derivatives(chebyshev_D(N), 6.0, N, None)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    M = np.log10(np.abs(D1) + 1e-6)
    im = ax.imshow(M, cmap="viridis")
    ax.set_title(r"First-derivative matrix $D_1$ ($25\times25$)")
    ax.set_xlabel("column $k$"); ax.set_ylabel("row $j$")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"$\log_{10}|D_{1,jk}|$", fontsize=13)
    ax.text(0.5, -0.16, "dense: every node couples to every other",
            transform=ax.transAxes, ha="center", fontsize=12, color=GREY)
    save(fig, "fig_diffmat.pdf")


def fig_blocks():
    fig, (aA, aB) = plt.subplots(1, 2, figsize=(11.0, 5.8))
    rows = ["continuity\n+ state", "x-momentum", "y-momentum", "energy"]
    cols = [r"$\hat u$", r"$\hat v$", r"$\hat T$", r"$\hat p$"]
    A_zero = {(3, 3)}                                        # energy row has no pressure term
    A_nz = {(i, j) for i in range(4) for j in range(4)} - A_zero
    B_nz = {(0, 2), (0, 3), (1, 0), (2, 1), (3, 2)}          # B holds the c-terms only
    # short structural tag per block (full expressions live in the doc table)
    labelsA = {
        (0, 0): r"$i\alpha I$",
        (0, 1): r"$D_1{-}\frac{\bar T'}{\bar T}$",
        (0, 2): r"$-i\alpha\frac{\bar U}{\bar T}$",
        (0, 3): r"$i\alpha\gamma M^2\bar U$",
        (1, 0): "$i\\alpha\\bar U$\n$+\\,$visc.",
        (1, 1): "$\\bar U'$\n$+\\,$visc.",
        (1, 2): "transport",
        (1, 3): r"$i\alpha\bar\rho^{-1}$",
        (2, 0): "visc.",
        (2, 1): "$i\\alpha\\bar U$\n$+\\,$visc.",
        (2, 2): "transport",
        (2, 3): r"$\bar\rho^{-1}D_1$",
        (3, 0): "dissip.",
        (3, 1): r"$\bar T'{+}\cdots$",
        (3, 2): "$i\\alpha\\bar U$\n$-\\,\\mathcal{C}_\\kappa\\mathcal{L}_c$",
        (3, 3): r"$\mathbf{0}$",
    }
    labelsB = {
        (0, 2): r"$-\frac{i\alpha}{\bar T}$", (0, 3): r"$i\alpha\gamma M^2$",
        (1, 0): r"$i\alpha$", (2, 1): r"$i\alpha$", (3, 2): r"$i\alpha$",
    }
    panels = ((aA, A_nz, labelsA, r"$A$  —  dense operator (15 blocks)", PM),
              (aB, B_nz, labelsB, r"$B$  —  the $c$-terms only (5 blocks)", REF))
    for ax, nz, labels, ttl, c in panels:
        for i in range(4):
            for j in range(4):
                on = (i, j) in nz
                ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=(c if on else "white"),
                             edgecolor=INK, lw=1.2, alpha=0.5 if on else 1.0))
                lab = labels.get((i, j))
                if lab:
                    ax.text(j + 0.5, 3 - i + 0.5, lab, ha="center", va="center",
                            fontsize=9.0, color=INK)
        for j, cl in enumerate(cols):
            ax.text(j + 0.5, 4.2, cl, ha="center", fontsize=14)
        for i, rl in enumerate(rows):
            ax.text(-0.14, 3 - i + 0.5, rl, ha="right", va="center", fontsize=14)
        ax.set_xlim(-2.2, 4.2); ax.set_ylim(-0.3, 4.7)
        ax.set_title(ttl, fontsize=16); ax.axis("off")
    fig.suptitle(r"$A\,\phi=c\,B\,\phi$: each $4n\times4n$ matrix is a "
                 r"$4\times4$ grid of $n\times n$ blocks (rows $=$ equations, columns $=$ fields)",
                 fontsize=16, y=1.03)
    fig.tight_layout()
    save(fig, "fig_blocks.pdf")


def fig_bc_rows():
    fig, ax = plt.subplots(figsize=(5.4, 6.2))
    n = 10
    fields = [(r"$\hat u$", COOL, "0", "0"),
              (r"$\hat v$", COOL, "0", "0"),
              (r"$\hat T$", HOT, "iso/adiab.", "0"),
              (r"$\hat p$", GREY, None, None)]
    for k, (lab, col, wall, free) in enumerate(fields):
        y0 = k * (n + 1)
        for r in range(n):
            yy = y0 + r
            isbc = (r in (0, n - 1)) and (wall is not None)
            ax.add_patch(Rectangle((0, yy), 1, 1,
                         facecolor=(REF if isbc else col), edgecolor="white",
                         lw=0.6, alpha=0.85 if isbc else 0.35))
        ax.text(1.2, y0 + n / 2, lab, fontsize=15, va="center")
        if wall is not None:
            ax.text(1.2, y0 + 0.5, f"freestream → {free}", fontsize=10.5, va="center", color=REF)
            ax.text(1.2, y0 + n - 0.5, f"wall → {wall}", fontsize=10.5, va="center", color=REF)
        else:
            ax.text(1.2, y0 + n / 2 - 1.4, "no explicit BC", fontsize=10.5, va="center", color=GREY)
    ax.set_xlim(-0.2, 4.0); ax.set_ylim(-1, 4 * (n + 1))
    ax.set_title("Boundary conditions = row replacement\nin the state vector $[\\hat u;\\hat v;\\hat T;\\hat p]$",
                 fontsize=14)
    ax.axis("off")
    save(fig, "fig_bc_rows.pdf")


def fig_evp_planes():
    ev, vec, y, p, d = we_solve(10.0, 180)
    di = we_disc_index(ev)
    cr, ci = ev.real, ev.imag
    k0 = int(np.argmax(ci))                     # largest c_i (spurious)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.4))
    # --- temporal: the REAL M=6 spectrum ---
    other = np.ones(len(ev), bool); other[di] = False
    a1.scatter(cr[other], ci[other], s=18, color=GREY, alpha=0.6, label="continuous-spectrum / damped")
    if k0 != di:
        a1.scatter([cr[k0]], [ci[k0]], s=80, facecolor="none", edgecolor=HOT, lw=2.0, zorder=4,
                   label=r"largest $c_i$ but spurious ($y_{\max}$-dependent)")
    a1.scatter([cr[di]], [ci[di]], s=170, color=PM, marker="*", edgecolor=INK, zorder=6,
               label=f"discrete Mack mode  $c={cr[di]:.3f}{ci[di]:+.3f}i$")
    a1.axhline(0, color=INK, lw=1.0)
    a1.text(0.46, 0.004, r"$c_i=0$ (neutral)", fontsize=10.5)
    a1.set_xlim(-0.05, 1.45); a1.set_ylim(-0.18, 0.055)
    a1.set_xlabel(r"$c_r=\operatorname{Re}(c)$"); a1.set_ylabel(r"$c_i=\operatorname{Im}(c)$")
    a1.set_title(r"Temporal: real $\alpha\!\to\!c$  (real $M{=}6$ spectrum)")
    a1.legend(loc="lower center", fontsize=8.6, framealpha=0.95); a1.grid(alpha=0.2)
    # --- spatial: deterministic schematic ---
    ar = np.array([0.10, 0.14, 0.17, 0.20, 0.24, 0.28, 0.31, 0.16, 0.19, 0.26])
    ai = np.array([0.10, 0.07, 0.05, 0.08, 0.06, 0.09, 0.11, 0.04, 0.07, 0.05])
    a2.scatter(ar, ai, s=24, color=GREY, alpha=0.7, label="other / damped roots")
    a2.scatter([0.205], [0.0], s=80, color=REF, marker="x", zorder=4, label=r"shift target $\alpha_0=\omega/c_{\rm ph}$")
    a2.scatter([0.205], [-0.02], s=150, color=PM, marker="*", edgecolor=INK, zorder=6, label="physical mode (growing)")
    a2.axhline(0, color=INK, lw=1.0); a2.set_ylim(-0.075, 0.15)
    a2.annotate(r"$\alpha_i<0:$ growth", xy=(0.205, -0.02), xytext=(0.045, -0.06),
                fontsize=11, color=PM, arrowprops=dict(arrowstyle="-|>", color=PM, lw=1.3))
    a2.set_xlabel(r"$\alpha_r=\operatorname{Re}(\alpha)$"); a2.set_ylabel(r"$\alpha_i=\operatorname{Im}(\alpha)$")
    a2.set_title(r"Spatial: real $\omega\!\to\!\alpha$  (schematic)")
    a2.legend(loc="upper right", fontsize=8.8); a2.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "fig_evp_planes.pdf")


def fig_neutral_field():
    import csv
    path = os.path.join(REPO, "verification/first_mode/_ozgen_refdigitize/secondmode_grid.csv")
    best = None
    for Ma in (6, 7, 8, 4, 10):
        Re, al, ci = [], [], []
        with open(path) as f:
            for r in csv.DictReader(f):
                if abs(float(r["Ma"]) - Ma) > 1e-6:
                    continue
                if r["c_i"] in ("", "nan") or r["resolved"] != "1":
                    continue
                Re.append(float(r["Re"])); al.append(float(r["alpha"])); ci.append(float(r["c_i"]))
        if len(ci) > 200:
            best = (Ma, np.array(Re), np.array(al), np.array(ci)); break
    Ma, Re, al, ci = best
    Ru = np.unique(Re); Au = np.unique(al)
    Z = np.full((len(Au), len(Ru)), np.nan)
    ri = {v: i for i, v in enumerate(Ru)}; ai = {v: i for i, v in enumerate(Au)}
    for R, a, c in zip(Re, al, ci):
        Z[ai[a], ri[R]] = c
    Zm = np.ma.masked_invalid(Z)
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    vmax = np.nanmax(np.abs(Zm))
    pc = ax.pcolormesh(Ru, Au, Zm, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="gouraud")
    Rc = None
    try:
        cs = ax.contour(Ru, Au, Zm, levels=[0.0], colors="k", linewidths=3.0)
        segs = [s for col in cs.allsegs for s in col if len(s) > 1]
        if segs:
            allpts = np.vstack(segs)
            Rc = allpts[:, 0].min()              # critical Reynolds number = nose of the curve
    except Exception:
        pass
    cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(r"temporal growth rate $c_i$", fontsize=13)
    ax.set_xlabel(r"$R=\sqrt{Re_x}$"); ax.set_ylabel(r"$\alpha_L$  (scaled wavenumber)")
    ax.set_title(f"Neutral curve = $\\{{c_i=0\\}}$ from the growth field ($M={Ma}$, 2nd mode)")
    ax.text(0.97, 0.05, "red: unstable ($c_i>0$)", transform=ax.transAxes,
            ha="right", fontsize=11.5, color=HOT)
    ax.text(0.62, 0.90, "branch II (upper)", transform=ax.transAxes, fontsize=11, color=INK)
    ax.text(0.62, 0.10, "branch I (lower)", transform=ax.transAxes, fontsize=11, color=INK)
    if Rc is not None:
        ax.annotate(r"$R_{\rm crit}$ (onset)", xy=(Rc, np.interp(Rc, [Ru.min(), Ru.max()], [Au.mean(), Au.mean()])),
                    xytext=(Rc + 0.18 * (Ru.max() - Ru.min()), Au.min() + 0.12 * (Au.max() - Au.min())),
                    fontsize=11, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3))
    save(fig, "fig_neutral_field.pdf")


def fig_nfactor():
    # Illustrative sigma(x) rescaled so the integrated N reaches the O(9-11)
    # range typically quoted for transition onset (e.g. Mack second-mode
    # cases); shape is schematic, not a literal solved case -- see caption.
    x = np.linspace(0, 12, 500)
    sigma = 2.55 * np.exp(-((x - 6.0) ** 2) / 5.5) - 0.17
    sig_pos = np.maximum(sigma, 0)
    N = np.cumsum(sig_pos) * (x[1] - x[0])
    fig, ax1 = plt.subplots(figsize=(7.6, 4.4))
    ax1.fill_between(x, 0, sig_pos, color=FILL, alpha=0.8)
    ax1.plot(x, sigma, color=PM, lw=2.6, label=r"growth $\sigma=-\alpha_i$")
    ax1.axhline(0, color=GREY, lw=0.9)
    ax1.set_xlabel(r"streamwise distance (or $R$)")
    ax1.set_ylabel(r"local growth $\sigma$", color=PM)
    ax1.tick_params(axis="y", labelcolor=PM)
    x0 = x[np.argmax(sig_pos > 0)]
    ax1.axvline(x0, color=GREY, ls=":", lw=1.4)
    ax1.text(x0 + 0.1, sig_pos.max() * 0.8, "lower\nneutral pt.", fontsize=11, color=GREY)
    ax2 = ax1.twinx()
    ax2.plot(x, N, color=INK, lw=2.8, label=r"$N=\int\max(\sigma,0)\,dR$")
    ax2.set_ylabel(r"$N$-factor", color=INK)
    ax2.axhline(N.max(), color=REF, ls="--", lw=1.6)
    ax2.text(0.3, N.max() + 0.15, f"$e^N$ amplification (here $N\\approx{N.max():.1f}$, "
             "illustrative)", fontsize=11.5, color=REF)
    ax1.set_title(r"$N$-factor: integrate growth $\to$ transition at $N\approx9$–$11$")
    fig.tight_layout()
    save(fig, "fig_nfactor.pdf")


def fig_workflow(single=False):
    """Workflow chart. single=True labels the stage brackets by the sections of
    the combined single document instead of by companion-document names."""
    fig, ax = plt.subplots(figsize=(8.4, 7.6))
    ax.set_xlim(0, 10.8); ax.set_ylim(4.9, 19.9); ax.axis("off")

    def box(cx, cy, w, h, title, sub, core=False):
        fc = "#eaf4ef" if core else "white"
        ec = PM if core else INK
        ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                     boxstyle="round,pad=0.08,rounding_size=0.12",
                     fc=fc, ec=ec, lw=2.2 if core else 1.5))
        ax.text(cx, cy + 0.24, title, ha="center", va="center", fontsize=15, fontweight="bold")
        ax.text(cx, cy - 0.40, sub, ha="center", va="center", fontsize=13.5, color="#111")

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=16, color="#555555", lw=2.0))

    # --- main spine -------------------------------------------------------
    SX, W, H = 4.6, 6.3, 1.45
    spine = [
        (19.0, "1 · Flow conditions", r"$M_e,\ Re,\ T_w/T_e$, gas model", False),
        (17.0, "2 · Mean flow", r"self-similar BVP $\to\ \overline{U},\overline{T}$ and $y$-derivatives", False),
        (15.0, "3 · Disturbance ODEs", r"linearise + normal mode $\to$ Eqs. (C)–(E)", False),
        (13.0, "4 · Discretisation", r"Chebyshev collocation:  $\mathrm{D}\to D_1,\ \ \mathrm{D}^2\to D_2$", False),
        (11.0, "5 · Eigenvalue problem", r"$4(n{+}1)\times4(n{+}1)$ pencil", True),
    ]
    for cy, t, s, c in spine:
        box(SX, cy, W, H, t, s, c)
    for i in range(len(spine) - 1):
        arrow(SX, spine[i][0] - 0.76, SX, spine[i + 1][0] + 0.76)

    # --- document attribution (right margin) ------------------------------
    def doc_tag(y_top, y_bot, label, color="#111"):
        x = 7.95
        ax.plot([x, x], [y_top, y_bot], color=color, lw=1.6)
        ax.plot([x - 0.09, x], [y_top, y_top], color=color, lw=1.6)
        ax.plot([x - 0.09, x], [y_bot, y_bot], color=color, lw=1.6)
        ax.text(x + 0.18, 0.5 * (y_top + y_bot), label, ha="left", va="center",
                fontsize=13, color=color, style="italic")
    if single:
        doc_tag(17.72, 16.28, "Section 2:\nmean flow")
        doc_tag(15.72, 12.28, "Section 3:\ndisturbance\nequations", color=PM)
        doc_tag(11.72, 10.28, "Section 4:\nnumerical\nmethods")
    else:
        doc_tag(17.72, 16.28, "mean-flow\ndocument")
        doc_tag(15.72, 12.28, "this document", color=PM)
        doc_tag(11.72, 10.28, "numerical-methods\ndocument")

    # --- temporal / spatial fork ------------------------------------------
    TX, PX = 2.35, 7.15
    arrow(SX - 0.9, 10.24, TX + 0.4, 9.35)
    arrow(SX + 0.9, 10.24, PX - 0.4, 9.35)
    ax.text(1.30, 9.9, r"temporal:  $\alpha\in\mathbb{R}$, solve $c$",
            fontsize=13, color="#111", ha="center", style="italic")
    ax.text(9.25, 9.9, r"spatial:  $\omega\in\mathbb{R}$, solve $\alpha$",
            fontsize=13, color="#111", ha="center", style="italic")
    box(TX, 8.6, 4.3, H, r"$A(\alpha)\,\phi=c\,B(\alpha)\,\phi$",
        r"temporal growth $\omega_i=\alpha c_i$")
    box(PX, 8.6, 4.6, H, r"$(C_0+\alpha C_1+\alpha^2C_2)\,\phi=0$",
        r"spatial growth $-\alpha_i$")
    arrow(TX, 7.84, TX, 6.86); arrow(PX, 7.84, PX, 6.86)
    box(TX, 6.1, 4.3, H, "Neutral curves", r"locus $c_i=0$ in $(R,\alpha)$", True)
    box(PX, 6.1, 4.6, H, r"$N$-factor", r"$N(R)=\int-\alpha_i\,\mathrm{d}R\ \to\ e^N$", True)
    save(fig, "fig_workflow_single.pdf" if single else "fig_workflow.pdf")


def fig_companion():
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    ax.axis("off"); ax.set_xlim(0, 12.2); ax.set_ylim(0.6, 6)
    ax.add_patch(FancyBboxPatch((0.15, 2.55), 3.25, 1.5,
                 boxstyle="round,pad=0.08,rounding_size=0.1", fc="white", ec=INK, lw=1.6))
    ax.text(1.78, 3.62, "Quadratic EVP", ha="center", fontsize=12.5, fontweight="bold")
    ax.text(1.78, 3.05, r"$(C_0+\alpha C_1+\alpha^2 C_2)\phi=0$", ha="center", fontsize=10.5)
    ax.text(1.78, 4.45, r"size $n$ (the $4n$ operator)", ha="center", fontsize=10.5, color=GREY)
    ax.add_patch(FancyArrowPatch((3.5, 3.3), (4.55, 3.3), arrowstyle="-|>", mutation_scale=16, color=GREY, lw=1.8))
    ax.text(4.02, 3.6, "companion", ha="center", fontsize=10, color=GREY)

    def pencil(x0, name, blocks):
        for i in range(2):
            for j in range(2):
                lab, fill = blocks.get((i, j), ("", False))
                ax.add_patch(Rectangle((x0 + j, 3.3 - i), 1, 1,
                             facecolor=(PM if fill else "white"), ec=INK, lw=1.2, alpha=0.5 if fill else 1.0))
                if lab:
                    ax.text(x0 + j + 0.5, 3.3 - i + 0.5, lab, ha="center", va="center", fontsize=11)
        ax.text(x0 + 1, 5.05, name, ha="center", fontsize=14, fontweight="bold")
    pencil(4.9, r"$\mathbf{L}$", {(0, 0): (r"$-C_1$", True), (0, 1): (r"$-C_0$", True),
                                  (1, 0): (r"$I$", True), (1, 1): ("$0$", False)})
    ax.text(7.55, 3.7, r"$=\ \alpha$", fontsize=16, ha="center")
    pencil(8.1, r"$\mathbf{R}$", {(0, 0): (r"$C_2$", True), (0, 1): ("$0$", False),
                                  (1, 0): ("$0$", False), (1, 1): (r"$I$", True)})
    ax.text(6.1, 1.75, r"acts on $[\,\psi;\ \phi\,]$ with auxiliary $\psi=\alpha\phi$;"
            r"  eigenvalue $=\alpha$,  $\phi$ is the lower half", ha="center", fontsize=11)
    ax.text(6.1, 1.15, r"size doubles:  $n\;\to\;2n$", ha="center", fontsize=12, color=REF, fontweight="bold")
    ax.set_title("Companion linearisation: a quadratic EVP becomes an ordinary EVP of double size", fontsize=14)
    save(fig, "fig_companion.pdf")


def fig_critlayer():
    p = make_flatplate_profile(4.0)
    yy = np.linspace(0, 6, 400); U = np.asarray(p(yy)["U"])
    cr = 0.436
    yc = yy[np.argmin(np.abs(U - cr))]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.3))
    a1.plot(U, yy, color=COOL, lw=2.6, label=r"$\overline{U}(y)$ (mean)")
    a1.axvline(cr, color=GREY, ls=":", lw=1.5)
    a1.scatter([cr], [yc], s=70, color=HOT, zorder=5)
    a1.annotate(r"critical layer  $\overline{U}=c_r$", xy=(cr, yc), xytext=(cr + 0.10, yc + 0.9),
                fontsize=11.5, color=HOT, arrowprops=dict(arrowstyle="-|>", color=HOT, lw=1.3))
    q = np.exp(-((yy - yc) ** 2) / 0.5) * np.exp(-0.12 * yy); q /= q.max()
    a1.plot(q, yy, color=PM, lw=2.2, label=r"$|\hat q(y)|$ (schematic)")
    a1.set_xlabel("mean velocity / amplitude"); a1.set_ylabel(r"wall-normal $y$")
    a1.set_title("Where the mode lives"); a1.legend(loc="upper right", fontsize=10.5); a1.grid(alpha=0.25)
    yd = np.linspace(0, 6, 400)
    disc = yd * np.exp(-1.1 * yd); disc /= disc.max()
    cont = 0.5 + 0.42 * np.sin(3.1 * yd); cont[yd < 0.25] = 0.0
    a2.plot(disc, yd, color=PM, lw=2.8, label="discrete mode\n(decays at edge)")
    a2.plot(cont, yd, color=REF, lw=2.0, ls="--", label="continuous spectrum\n(persists at edge)")
    a2.set_xlabel(r"$|\hat q(y)|$"); a2.set_ylabel(r"wall-normal $y$")
    a2.set_title("Physical vs spurious"); a2.legend(loc="upper right", fontsize=9.5); a2.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig_critlayer.pdf")


def fig_eigenfunctions():
    ev, vec, y, p, d = we_solve(10.0, 180)
    di = we_disc_index(ev); n = len(y)
    order = np.argsort(y); ya = y[order]
    specs = [(r"$|\hat u|$", COOL), (r"$|\hat v|$", "#7b3fa0"), (r"$|\hat T|$", HOT), (r"$|\hat p|$", PM)]
    arrs = [np.abs(vec[k * n:(k + 1) * n, di])[order] for k in range(4)]
    arrs = [a / (a.max() + 1e-30) for a in arrs]
    env = np.maximum.reduce(arrs)
    ycut = (ya[env > 0.02].max() * 1.25) if (env > 0.02).any() else ya.max()
    U = np.asarray(p(ya / d)["U"]); cr = ev[di].real
    yc = ya[np.argmin(np.abs(U - cr))]
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for (lab, col), a in zip(specs, arrs):
        ax.plot(a, ya, color=col, lw=2.4, label=lab)
    ax.axhline(yc, color=GREY, ls=":", lw=1.5)
    ax.text(0.42, yc + 0.04 * ycut, r"critical layer $\overline{U}=c_r$", fontsize=10.5, color=GREY)
    ax.set_ylim(0, ycut); ax.set_xlim(0, 1.05)
    ax.set_xlabel("normalised amplitude"); ax.set_ylabel(r"wall-normal $y$  (in $L^*$)")
    ax.set_title(r"Computed eigenfunction: $M{=}6$ Mack mode, $c=0.930+0.020\,i$")
    ax.legend(loc="upper right"); ax.grid(alpha=0.25)
    ax.text(0.5, 0.015 * ycut, "wall-confined pressure $\\to$ acoustic 2nd mode", fontsize=9.5, color="#555")
    save(fig, "fig_eigenfunctions.pdf")


def fig_acoustic_trapping():
    ev, vec, y, p, d = we_solve(10.0, 180)
    di = we_disc_index(ev); cr = ev[di].real
    order = np.argsort(y); ya = y[order]
    bf = p(ya / d); U = np.asarray(bf["U"]); T = np.asarray(bf["T"]); Me = WE["Ma"]
    a = np.sqrt(T) / Me
    Mrel = (U - cr) / a
    ycrit = ya[np.argmin(np.abs(Mrel))]
    ysonic = ya[np.argmin(np.abs(Mrel + 1.0))]
    ycut = max(ycrit * 1.8, ysonic * 1.5)
    m = ya <= ycut
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.axhspan(0, ysonic, color=FILL, alpha=0.8, zorder=0)
    ax.plot(Mrel[m], ya[m], color=INK, lw=2.8, zorder=3)
    ax.axvline(-1, color=REF, ls="--", lw=1.6); ax.axvline(0, color=GREY, ls=":", lw=1.4)
    ax.annotate(r"relative sonic line $|M_{\rm rel}|{=}1$", xy=(-1, ysonic),
                xytext=(-2.3, ysonic + ycut * 0.10), color=REF, fontsize=10.5,
                arrowprops=dict(arrowstyle="-|>", color=REF, lw=1.2))
    ax.annotate(r"critical layer $M_{\rm rel}{=}0$", xy=(0, ycrit),
                xytext=(0.12, ycrit + ycut * 0.06), color="#555", fontsize=10.5,
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.2))
    ax.text(-0.5, ysonic * 0.5, "trapped\nacoustic region\n(2nd-mode cavity)",
            color=PM, fontsize=10.5, fontweight="bold", va="center", ha="center")
    ax.set_xlabel(r"relative Mach number  $M_{\rm rel}(y)=(\overline{U}-c_r)/\overline{a}$")
    ax.set_ylabel(r"wall-normal $y$  (in $L^*$)")
    ax.set_ylim(0, ycut)
    ax.set_title(r"Mack mode = sound trapped between wall and sonic line ($M_e{=}6$)")
    ax.grid(alpha=0.25)
    save(fig, "fig_acoustic_trapping.pdf")


def fig_convergence():
    p = make_flatplate_profile(WE["Ma"]); d = delta_star_over_lstar(p)
    def c_at(N):
        ev, _, _ = solve_temporal_2d(p, WE["al"], WE["Re"], WE["Ma"],
                                           N=N, y_max=10 * d, length_scale="L_star")
        return ev[np.argmin(np.abs(ev - WE["c_disc"]))]
    cref = c_at(200)
    Ns = [20, 30, 40, 60, 80, 100, 120, 140, 160]
    err = [abs(c_at(N) - cref) for N in Ns]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogy(Ns, err, "o-", color=PM, lw=2.4, ms=7)
    ax.axvline(128, color=GREY, ls=":", lw=1.5)
    ax.text(131, err[0] * 0.5, r"default $N{=}128$", fontsize=10.5, color=GREY)
    ax.set_xlabel("collocation points $N$")
    ax.set_ylabel(r"eigenvalue error $|c(N)-c(200)|$")
    ax.set_title("Spectral (exponential) convergence of the Mack-mode eigenvalue")
    ax.grid(alpha=0.3, which="both")
    save(fig, "fig_convergence.pdf")


def fig_transition_roadmap():
    fig, ax = plt.subplots(figsize=(9.8, 2.5)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    stages = [("Receptivity", "disturbances\nenter the BL"),
              ("Linear modal\ngrowth (LST)", "THIS DOCUMENT"),
              ("Secondary\ninstability", "2D $\\to$ 3D"),
              ("Nonlinear\nbreakdown", ""), ("Turbulence", "")]
    w, gap, x = 1.66, 0.27, 0.2
    for i, (t, s) in enumerate(stages):
        core = (i == 1)
        ax.add_patch(FancyBboxPatch((x, 0.5), w, 1.0, boxstyle="round,pad=0.04,rounding_size=0.08",
                     fc=("#eaf4ef" if core else "white"), ec=(PM if core else GREY), lw=2.6 if core else 1.3))
        ax.text(x + w / 2, 1.12, t, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=(INK if core else "#555"))
        if s:
            ax.text(x + w / 2, 0.74, s, ha="center", va="center", fontsize=8.4,
                    color=(PM if core else "#777"), fontweight=("bold" if core else "normal"))
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + w, 1.0), (x + w + gap, 1.0), arrowstyle="-|>",
                         mutation_scale=13, color=GREY, lw=1.7))
        x += w + gap
    ax.set_title("Where LST sits: this document covers the highlighted (linear-growth) stage", fontsize=13)
    save(fig, "fig_transition_roadmap.pdf")


def fig_split():
    """Step 1/2 of linearisation: total = mean (steady) + small disturbance.

    Uses the real Ma=4.5 flat-plate mean profile (same data source as
    fig_baseflow) for Ubar(y); the disturbance wiggle is a schematic
    small-amplitude wave, drawn at ~5% of the mean to make the ordering
    |q'| << |qbar| explicit, as invoked in the Step 2 linearisation text.
    """
    p = make_flatplate_profile(4.5)
    y = np.linspace(0, 6, 500)
    Ubar = np.asarray(p(y)["U"])
    amp = 0.05
    yq = 3.6
    wiggle = amp * np.exp(-((y - yq) ** 2) / 1.4) * np.sin(2 * np.pi * y / 0.9)
    total = Ubar + wiggle

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 5.0), sharey=True)

    a1.plot(Ubar, y, color=COOL, lw=2.8)
    a1.set_xlabel(r"$\overline{U}(y)$")
    a1.set_ylabel(r"wall-normal  $y$")
    a1.set_title("mean (steady)\n" r"$\overline{q}(y)$", fontsize=15, pad=10)
    a1.set_xlim(0, 1.16)
    a1.grid(alpha=0.25)

    a2.plot(Ubar, y, color=COOL, lw=1.8, ls="--", alpha=0.55, label=r"mean $\overline{q}$")
    a2.plot(total, y, color=INK, lw=2.6, label=r"total $q=\overline{q}+q'$")
    a2.fill_betweenx(y, Ubar, total, color=PM, alpha=0.35, lw=0, label=r"disturbance $q'$")
    a2.annotate(r"$|q^\prime|\ll|\overline{q}|$", xy=(total[np.argmin(np.abs(y - yq))], yq),
                xytext=(0.68, 5.3), fontsize=13, color=PM,
                arrowprops=dict(arrowstyle="-|>", color=PM, lw=1.3))
    a2.set_xlabel(r"$q(x,y,t)$")
    a2.set_title("total = mean + disturbance (Step 1)\n"
                 r"$q=\overline{q}(y)+q^\prime(x,y,t)$", fontsize=15, pad=10)
    a2.set_xlim(0, 1.16)
    a2.legend(loc="lower right", fontsize=11.5, framealpha=0.95)
    a2.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle("Linearisation Step 1/2: split, then drop products of primes", fontsize=16, y=0.99)
    save(fig, "fig_split.pdf")


def fig_shooting_scheme():
    """Schematic of Part B's shooting method (Sections 10-14 of
    numerical_methods.tex): eigen-decompose the freestream A_infty, keep the
    3 decaying columns, QR-orthonormalise, march by RK4 down to the wall
    (re-orthonormalising along the way), build the 3x3 wall matrix M(c),
    and let Nelder-Mead search on c to drive sigma_min(M) -> 0.
    """
    fig, ax = plt.subplots(figsize=(8.2, 9.4))
    ax.axis("off"); ax.set_xlim(-0.3, 11.6); ax.set_ylim(-1.6, 16.4)

    y_top, y_bot = 14.4, 1.6
    x_axis = 1.1
    ax.annotate("", xy=(x_axis, y_bot), xytext=(x_axis, y_top),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.0))
    ax.text(x_axis - 0.35, y_top + 0.35, r"$y_{\max}$", fontsize=13, ha="center")
    ax.text(x_axis - 0.35, y_bot - 0.35, r"wall $y=0$", fontsize=13, ha="center")
    ax.text(x_axis - 0.75, (y_top + y_bot) / 2, r"$y$", fontsize=14, rotation=90, va="center")

    x_mid = 6.15

    # top: eigen-decomposition box
    ax.add_patch(FancyBboxPatch((2.3, y_top - 0.55), 7.7, 1.15,
                 boxstyle="round,pad=0.08,rounding_size=0.12", fc="#eaf4ef", ec=PM, lw=2.0))
    ax.text(x_mid, y_top + 0.16, r"eigendecompose pointwise $A_\infty(\alpha,c)$",
            ha="center", fontsize=12.5, fontweight="bold")
    ax.text(x_mid, y_top - 0.32, "keep the 3 decaying eigenvectors  $\\to$  QR (orthonormal columns of $Y$)",
            ha="center", fontsize=11, color="#333")

    # three marching columns, clear of the y-axis line and the RK4 labels
    cols_x = [4.0, 6.15, 8.3]
    labels = [r"$Y^{(1)}$", r"$Y^{(2)}$", r"$Y^{(3)}$"]
    n_steps = 6
    y_march_top, y_march_bot = y_top - 1.35, y_bot + 1.0
    ys = np.linspace(y_march_top, y_march_bot, n_steps)
    for cx, lab in zip(cols_x, labels):
        xs = cx + 0.18 * np.sin(np.linspace(0, 3.0, n_steps)) * np.linspace(0, 1, n_steps)
        ax.plot(xs, ys, color=COOL, lw=1.8, alpha=0.85)
        for k in range(n_steps - 1):
            ax.add_patch(FancyArrowPatch((xs[k], ys[k] - 0.05), (xs[k + 1], ys[k + 1] + 0.05),
                         arrowstyle="-|>", mutation_scale=9, color=COOL, lw=1.1, alpha=0.85))
        ax.text(cx, y_march_top + 0.32, lab, fontsize=13, color=COOL, ha="center")

    for k in range(1, n_steps - 1):
        yk = ys[k]
        ax.plot([3.3, 9.0], [yk, yk], color=GREY, lw=0.9, ls=":", alpha=0.6, zorder=0)
        ax.text(9.15, yk, "RK4 step +\nQR re-orthon.", fontsize=9.0, color=GREY, va="center")

    # wall matrix box
    y_wall_box = y_bot - 1.05
    ax.add_patch(FancyBboxPatch((2.3, y_wall_box - 0.5), 7.7, 1.0,
                 boxstyle="round,pad=0.08,rounding_size=0.12", fc="#fbe9e4", ec=REF, lw=2.0))
    ax.text(x_mid, y_wall_box + 0.20, r"wall matrix  $M(c)=[\,Y_1(0);\,Y_3(0);\,Y_5(0)\,]$  ($3\times3$)",
            ha="center", fontsize=12.2, fontweight="bold", color="#8a2f0d")
    ax.text(x_mid, y_wall_box - 0.28, r"rows $1,3,5=(u,v,T)$ at the wall  $\Rightarrow$  $\sigma_{\min}(M(c))=0$",
            ha="center", fontsize=11, color="#8a2f0d")

    # outer Nelder-Mead loop, placed well below the wall-matrix box
    y_nm_box = y_wall_box - 2.3
    ax.add_patch(FancyBboxPatch((2.3, y_nm_box - 0.5), 7.7, 1.0,
                 boxstyle="round,pad=0.08,rounding_size=0.1", fc="white", ec=INK, lw=1.6))
    ax.text(x_mid, y_nm_box, r"Nelder–Mead updates $c=c_r+\mathrm{i}c_i$ ($\alpha$ fixed)",
            ha="center", fontsize=11.6, fontweight="bold")
    ax.text(x_mid, y_nm_box - 0.42, r"to drive $\sigma_{\min}\to0$", ha="center", fontsize=10.6, color="#333")

    ax.add_patch(FancyArrowPatch((x_mid, y_wall_box - 0.5), (x_mid, y_nm_box + 0.5),
                 arrowstyle="-|>", mutation_scale=14, color="#8a2f0d", lw=1.6))

    # feedback path: NM box -> right side -> back up to the eigendecomposition box
    x_loop = 10.5
    ax.add_patch(FancyArrowPatch((10.0, y_nm_box), (x_loop, y_nm_box),
                 arrowstyle="-", color=INK, lw=1.4))
    ax.add_patch(FancyArrowPatch((x_loop, y_nm_box), (x_loop, y_top),
                 arrowstyle="-", color=INK, lw=1.4))
    ax.add_patch(FancyArrowPatch((x_loop, y_top), (10.0, y_top),
                 arrowstyle="-|>", mutation_scale=14, color=INK, lw=1.4))
    ax.text(x_loop + 0.30, (y_top + y_nm_box) / 2, "updated $c$ re-seeds the\nnext eigendecomposition",
            fontsize=10.0, color=INK, rotation=90, va="center", ha="center")

    ax.add_patch(FancyArrowPatch((x_mid, y_top - 0.55), (x_mid, y_march_top + 0.55),
                 arrowstyle="-|>", mutation_scale=14, color=PM, lw=1.6))
    ax.add_patch(FancyArrowPatch((x_mid, y_march_bot - 0.05), (x_mid, y_wall_box + 0.5),
                 arrowstyle="-|>", mutation_scale=14, color=REF, lw=1.6))

    save(fig, "fig_shooting_scheme.pdf")


def _assemble_temporal_AB_small(Ma, alpha, Re, N, y_max=6.0, wall_bc="isothermal"):
    """Real A, B assembly at a small N, mirroring pymack/temporal_solver.py's
    solve_temporal_2d block-for-block (same equations, same row order),
    just returning the matrices instead of solving the EVP. This is NOT a
    reimplementation of new physics -- every line below is copied from
    pymack/temporal_solver.py (Ozgen & Kircali 2008 arrangement) so the
    resulting A, B are numerically real, code-accurate matrices.
    """
    from pymack.equations import transport_conductivity_data, transport_temperature_derivatives
    from pymack.scales import delta_star_over_lstar, rescale_baseflow_derivatives
    from pymack.solver import temperature_wall_operator
    import numpy as _np

    gamma, Pr = 1.4, 0.72
    baseflow = make_flatplate_profile(Ma)
    D_eta = chebyshev_D(N)
    y, D1, D2 = physical_derivatives(D_eta, y_max, N, None)
    bf = baseflow(y)

    n = len(y)
    I = _np.eye(n)
    ia = 1j * alpha
    a2 = alpha ** 2

    U = _np.asarray(bf['U']); dU = _np.asarray(bf['dU']); d2U = _np.asarray(bf['d2U'])
    T = _np.asarray(bf['T']); dT = _np.asarray(bf['dT']); d2T = _np.asarray(bf['d2T'])
    rho = _np.asarray(bf['rho']); mu = _np.asarray(bf['mu']); dmu = _np.asarray(bf['dmu'])
    dmu_dT_v, d2mu_dT2_v = transport_temperature_derivatives(bf)
    kappa_v, _dk, dkappa_dT_v, d2kappa_dT2_v, _ = transport_conductivity_data(bf, Pr)
    pr_local = _np.asarray(bf.get('Pr_local', _np.full_like(T, Pr)))

    Ub = _np.diag(U); dUb = _np.diag(dU); d2Ub = _np.diag(d2U)
    dTb = _np.diag(dT); d2Tb = _np.diag(d2T)
    rhob_inv = _np.diag(1.0 / rho); Tb_inv = _np.diag(1.0 / T)
    mub = _np.diag(mu); dmub = _np.diag(dmu)
    dmu_dT = _np.diag(dmu_dT_v); d2mu_dT2 = _np.diag(d2mu_dT2_v)
    kappab_inv = _np.diag(1.0 / kappa_v)
    dkappa_dT = _np.diag(dkappa_dT_v); d2kappa_dT2 = _np.diag(d2kappa_dT2_v)

    visc = rhob_inv / Re
    cond = _np.diag(gamma * mu / (pr_local * Re * rho))
    diss = _np.diag(gamma * (gamma - 1.0) * Ma ** 2 / (Re * rho))

    nn = 4 * n
    A = _np.zeros((nn, nn), dtype=complex)
    B = _np.zeros((nn, nn), dtype=complex)

    def blk(i, j):
        return (slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n))

    A[blk(0, 0)] = ia * I
    A[blk(0, 1)] = D1 - Tb_inv @ dTb
    A[blk(0, 2)] = -ia * Ub @ Tb_inv
    A[blk(0, 3)] = ia * gamma * Ma ** 2 * Ub
    B[blk(0, 2)] = -ia * Tb_inv
    B[blk(0, 3)] = ia * gamma * Ma ** 2 * I

    A[blk(1, 0)] = ia * Ub - visc @ (mub @ D2 + dmub @ D1) + (4.0 / 3.0) * a2 * visc @ mub
    A[blk(1, 1)] = dUb - ia * visc @ ((1.0 / 3.0) * mub @ D1 + dmub)
    A[blk(1, 2)] = -visc @ (dmu_dT @ dUb @ D1 + dmu_dT @ d2Ub + d2mu_dT2 @ dUb @ dTb)
    A[blk(1, 3)] = ia * rhob_inv
    B[blk(1, 0)] = ia * I

    A[blk(2, 0)] = -ia * visc @ ((1.0 / 3.0) * mub @ D1 - (2.0 / 3.0) * dmub)
    A[blk(2, 1)] = ia * Ub - visc @ ((4.0 / 3.0) * mub @ D2 + (4.0 / 3.0) * dmub @ D1) + a2 * visc @ mub
    A[blk(2, 2)] = -ia * visc @ (dmu_dT @ dUb)
    A[blk(2, 3)] = rhob_inv @ D1
    B[blk(2, 1)] = ia * I

    conduction_operator = (D2 - a2 * I
                            + kappab_inv @ dkappa_dT @ (2.0 * dTb @ D1 + d2Tb)
                            + kappab_inv @ d2kappa_dT2 @ dTb @ dTb)
    A[blk(3, 0)] = (gamma - 1.0) * rhob_inv @ (ia * I) - diss @ (2.0 * mub @ dUb @ D1)
    A[blk(3, 1)] = dTb + (gamma - 1.0) * rhob_inv @ D1 - diss @ (2j * alpha * mub @ dUb)
    A[blk(3, 2)] = ia * Ub - cond @ conduction_operator - diss @ (dmu_dT @ dUb @ dUb)
    B[blk(3, 2)] = ia * I

    wall, free = n - 1, 0
    for var in range(2):
        for loc in (wall, free):
            row = var * n + loc
            A[row, :] = 0.0; B[row, :] = 0.0; A[row, row] = 1.0
    temp_slice = slice(2 * n, 3 * n)
    temp_wall_row = 2 * n + wall; temp_free_row = 2 * n + free
    A[temp_wall_row, :] = 0.0; B[temp_wall_row, :] = 0.0
    A[temp_wall_row, temp_slice] = temperature_wall_operator(D1, n, wall_bc)
    A[temp_free_row, :] = 0.0; B[temp_free_row, :] = 0.0
    A[temp_free_row, temp_free_row] = 1.0
    return A, B, n


def fig_AB_real():
    """A real, numerically-assembled A, B pair (small N) as a companion to
    the symbolic fig_blocks.pdf: same worked-example parameters (Ma=6,
    Re=5500, alpha=0.174) but N=6 (n=7, so 4n=28) to keep every block
    visually legible. Values come straight from the copied assembly in
    _assemble_temporal_AB_small(), which mirrors solve_temporal_2d().
    """
    N = 6
    A, B, n = _assemble_temporal_AB_small(WE["Ma"], WE["al"], WE["Re"], N=N, y_max=6.0)
    nn = 4 * n
    fig, (aA, aB) = plt.subplots(1, 2, figsize=(12.0, 5.6))
    cols = [r"$\hat u$", r"$\hat v$", r"$\hat T$", r"$\hat p$"]
    rows = ["C", "X", "Y", "E"]
    for ax, M, ttl in ((aA, A, r"$A$"), (aB, B, r"$B$")):
        Mm = np.log10(np.abs(M) + 1e-8)
        im = ax.imshow(Mm, cmap="viridis", vmin=-6, vmax=1, aspect="equal")
        for k in range(1, 4):
            ax.axhline(k * n - 0.5, color="white", lw=1.1)
            ax.axvline(k * n - 0.5, color="white", lw=1.1)
        ticks = [k * n + n / 2 - 0.5 for k in range(4)]
        ax.set_xticks(ticks); ax.set_xticklabels(cols, fontsize=14)
        ax.set_yticks(ticks); ax.set_yticklabels(rows, fontsize=14)
        ax.set_xlabel("field (column block)", fontsize=13)
        ax.set_ylabel("equation (row block)", fontsize=13)
        ax.set_title(f"{ttl}  ({nn}$\\times${nn}, $N{{=}}{N}$)", fontsize=16)
    cb = fig.colorbar(im, ax=[aA, aB], fraction=0.025, pad=0.03)
    cb.set_label(r"$\log_{10}|{\cdot}|$", fontsize=13)
    fig.suptitle("Real assembled operator: worked example ($M{=}6$, $R{=}5500$, "
                 r"$\alpha_L{=}0.174$) — complements the symbolic block figure",
                 fontsize=14.5, y=1.02)
    save(fig, "fig_AB_real.pdf")


if __name__ == "__main__":
    figs = [fig_setup, fig_ansatz, fig_baseflow, fig_cheb_grid, fig_diffmat,
            fig_blocks, fig_bc_rows, fig_companion, fig_critlayer, fig_evp_planes,
            fig_neutral_field, fig_nfactor, fig_workflow,
            fig_eigenfunctions, fig_acoustic_trapping, fig_convergence, fig_transition_roadmap,
            fig_split, fig_shooting_scheme, fig_AB_real]
    ok = 0
    for fn in figs:
        try:
            fn(); ok += 1
        except Exception as e:
            print("FAILED", fn.__name__, "->", repr(e))
    print(f"DONE {ok}/{len(figs)} figures")
