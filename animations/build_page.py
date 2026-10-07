"""Pack a chain written by run_chain.py into the self-contained HTML page.

    python animations/build_page.py [chain_nmax10.npz]

Writes site/index.html, the page published by GitHub Pages. The page compares the
chain with the Set A result of the paper at the same point, read from
reproduction/data/setA_points.csv, and with the one-loop prediction.
"""
from __future__ import annotations

import base64
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import gamma, gammaln, lpmv

HERE = Path(__file__).resolve().parent
NT_BIG = 72          # texture rows of the large sphere (small ones use every 2nd)



def paper_reference(n_max, lam4, m2, vol):
    """Set A result of the paper at this point, and the one-loop prediction for <c_0^2>."""
    from lattice_ds.perturbation import one_loop_zero_mode_moments
    with open(HERE.parent / "reproduction" / "data" / "setA_points.csv") as fh:
        row = [r for r in csv.DictReader(fh)
               if int(r["n_max"]) == n_max and abs(float(r["lam"]) - lam4) < 1e-12][0]
    return {"c0_2": float(row["phibar2"]) * vol, "c0_2_err": float(row["phibar2_err"]) * vol,
            "binder": float(row["U"]), "binder_err": float(row["U_err"]),
            "one_loop_c0_2": one_loop_zero_mode_moments(n_max, m2, lam4, vol)["c0_2"],
            "n_cfg": int(row["n_chains"]) * int(row["n_meas_per_chain"])}


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def plm_table(n_max, nt):
    """N_lm P_l^|m|(cos theta_i), same convention as harmonics.real_sph_harm
    (sqrt(2) included for m != 0). Shape (nt, (n_max+1)^2), index l*l+l+m."""
    th = (np.arange(nt) + 0.5) * np.pi / nt
    out = np.zeros((nt, (n_max + 1) ** 2), dtype=np.float32)
    for l in range(n_max + 1):
        for m in range(-l, l + 1):
            am = abs(m)
            k = np.sqrt((2 * l + 1) / (4 * np.pi)
                        * np.exp(gammaln(l - am + 1) - gammaln(l + am + 1)))
            out[:, l * l + l + m] = (1.0 if m == 0 else np.sqrt(2.0)) * k * lpmv(am, l, np.cos(th))
    return out


def binder_sd_coeff(seed=7, n=1000, reps=4000):
    """sd(U_N) * sqrt(N) for independent draws from exp(-x^4)."""
    rng = np.random.default_rng(seed)
    # x^4 ~ Gamma(1/4): x = +- g^(1/4)
    g = rng.gamma(0.25, size=(reps, n)) ** 0.25
    m2 = (g**2).mean(1)
    m4 = (g**4).mean(1)
    return float(np.std(1 - m4 / (3 * m2**2)) * np.sqrt(n))


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "chain_nmax10.npz"
    d = np.load(HERE / src)
    n_max = int(d["n_max"])
    vol = float(d["vol"])
    lam4 = float(d["lam4"])
    K = d["slices"].shape[0]
    lidx = np.array([l for l in range(n_max + 1) for _ in range(2 * l + 1)])

    sl = np.concatenate([d["start_slices"][None], d["slices"]]).astype(np.float64)  # (K+1,5,ns2)
    pb_post = np.concatenate([[float(d["start_c0"])], d["c0_post"]]) / np.sqrt(vol)
    pb_pre = np.concatenate([[float(d["start_c0"])], d["c0_pre"]]) / np.sqrt(vol)
    # remove the constant mode: on every slice it is phibar * sqrt(4 pi) in b_00
    sl[:, :, 0] -= (pb_post * np.sqrt(4 * np.pi))[:, None]

    scales = np.array([4.5 * np.sqrt(np.mean(sl[:, :, lidx == l] ** 2)) for l in range(n_max + 1)])
    q = np.clip(np.round(sl / scales[lidx] * 127), -127, 127).astype(np.int8)
    err = np.sqrt(np.mean((q * scales[lidx] / 127 - sl) ** 2)) / np.sqrt(np.mean(sl**2))
    print(f"quantisation rel. rms error {err:.4f}")

    # colour range from the actual field values on the great sphere
    plm = plm_table(n_max, NT_BIG)
    rms_nz = np.sqrt(np.mean(np.sum(sl[:, 0] ** 2, axis=1)) / (4 * np.pi))
    rms_tot = np.sqrt(rms_nz**2 + np.mean(pb_post**2))
    vmax = float(np.round(2.4 * rms_tot, 1))
    print(f"rms nonzero {rms_nz:.3f} total {rms_tot:.3f} vmax {vmax}")

    PROD = paper_reference(n_max, lam4, float(d["m2"]), vol)
    kurt = gamma(0.25) * gamma(1.25) / gamma(0.75) ** 2
    meta = dict(
        n_max=n_max, lam4=lam4, m2=float(d["m2"]), K=int(K), vol=vol,
        n_modes=int(d["n_modes"]), grid_points=int(d["grid_points"]),
        eps=float(d["eps"]), n_steps=int(d["n_steps"]), zmu=int(d["zmu"]),
        n_therm=int(d["n_therm"]), seed=int(d["seed"]),
        chi_deg=[float(x) for x in d["chi_deg"]], vmax=vmax, nt=NT_BIG,
        acceptance=float(np.mean(d["acc"])),
        exp_negdH=float(np.mean(np.exp(-d["dH"]))),
        pred_var=PROD["one_loop_c0_2"] / vol, pred_binder=float(1 - kurt / 3),
        kurt=float(kurt), binder_sd=binder_sd_coeff(),
        prod_var=PROD["c0_2"] / vol, prod_var_err=PROD["c0_2_err"] / vol,
        prod_binder=PROD["binder"], prod_binder_err=PROD["binder_err"],
        prod_n=PROD["n_cfg"],
    )
    import bands
    bN, qv, qu, _, _ = bands.bands(int(K), meta["pred_var"])
    meta["band_N"] = [int(x) for x in bN]
    meta["band_var"] = [[round(float(x), 4) for x in row] for row in qv]
    meta["band_U"] = [[round(float(x), 4) for x in row] for row in qu]
    data = dict(
        meta=meta,
        slices=b64(q), scales=[float(s) for s in scales],
        pb_pre=b64(pb_pre.astype(np.float32)), pb_post=b64(pb_post.astype(np.float32)),
        acc=b64(np.concatenate([[1], d["acc"]]).astype(np.uint8)),
        T=b64(np.concatenate([[1.5], d["T"]]).astype(np.float32)),
        shell=b64(d["shell"].astype(np.float32)),
        plm=b64(plm),
    )
    blob = json.dumps(data, separators=(",", ":"))
    body = (HERE / "template.html").read_text().replace("__DATA__", blob)
    out = HERE / "site"
    out.mkdir(exist_ok=True)
    # the template starts with the <title> and font <link>, which belong in <head>
    cut = body.index("<style>")
    (out / "index.html").write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<meta name=\"description\" content=\"Hybrid Monte Carlo sampling of lambda phi^4 theory "
        "on Euclidean de Sitter space (S^4): an animation of real configurations.\">\n"
        + body[:cut] +
        "<style>html{color-scheme:light}body{margin:0}[hidden]{display:none!important}</style>\n"
        "</head>\n<body>\n" + body[cut:] + "\n</body>\n</html>\n")
    print(f"wrote animations/site/index.html  {len(body)/1e6:.2f} MB   K={K}")


if __name__ == "__main__":
    main()
