# deep_test.py
# Usage: python deep_test.py input.qi.zip output.tsv

import sys
import zipfile
import io
import numpy as np
import struct
import re
from scipy import interpolate
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

if __name__ == "__main__":

    IN_FILE = sys.argv[1]
    OUT_FILE = sys.argv[2]

    N_Z_PTS = 256
    fcs = []

    print("Reading data from", IN_FILE)

    with zipfile.ZipFile(IN_FILE) as zf:

        with zf.open("shared-data/header.properties") as f:
            f.readline()
            sh_param = {}
            for b_line in f:
                s_line = b_line.decode("utf-8")
                m = re.search("=", s_line)
                sh_param[s_line[:m.end()-1]] = s_line[m.end():]

        ch = {}
        i = 0
        while True:
            try:
                name = sh_param[f"lcd-info.{i}.channel.name"][:-1]
                ch[name] = i
                i += 1
            except KeyError:
                break

        ch_num = ch["measuredHeight"]
        z_offset1 = float(sh_param[f"lcd-info.{ch_num}.encoder.scaling.offset"])
        z_multi1  = float(sh_param[f"lcd-info.{ch_num}.encoder.scaling.multiplier"])
        z_offset2 = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.nominal.scaling.offset"])
        z_multi2  = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.nominal.scaling.multiplier"])

        ch_num = ch["vDeflection"]
        f_offset1 = float(sh_param[f"lcd-info.{ch_num}.encoder.scaling.offset"])
        f_multi1  = float(sh_param[f"lcd-info.{ch_num}.encoder.scaling.multiplier"])
        f_offset2 = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.distance.scaling.offset"])
        f_multi2  = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.distance.scaling.multiplier"])
        f_offset3 = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.force.scaling.offset"])
        f_multi3  = float(sh_param[f"lcd-info.{ch_num}.conversion-set.conversion.force.scaling.multiplier"])

        with zf.open("header.properties") as f:
            f.readline()
            qi_param = {}
            for b_line in f:
                s_line = b_line.decode("utf-8")
                m = re.search("=", s_line)
                qi_param[s_line[:m.end()-1]] = s_line[m.end():]

        x_pix = int(qi_param["quantitative-imaging-map.position-pattern.grid.ilength"])
        y_pix = int(qi_param["quantitative-imaging-map.position-pattern.grid.jlength"])
        xy_pix = x_pix * y_pix

        z_pix = 0
        for i in range(xy_pix):
            with zf.open(f"index/{i}/header.properties") as f:
                for b_line in f:
                    s_line = b_line.decode("utf-8")
                    m = re.search(
                        "quantitative-imaging-series.header.quantitative-imaging-settings.extend.num-points=",
                        s_line
                    )
                    if m is not None:
                        z_pix = max(z_pix, int(s_line[m.end():]))

        fmap = np.empty((x_pix, y_pix, z_pix, 2))

        for j in range(y_pix):
            for i in range(x_pix):
                with zf.open(f"index/{i+j*x_pix}/segments/0/channels/measuredHeight.dat") as f:
                    b = f.read()
                    fh = io.BytesIO(b)
                    n_pts = len(b)//4

                    for k in range(n_pts):
                        raw = struct.unpack(">i", fh.read(4))[0]
                        z = z_multi1 * raw + z_offset1
                        z = z_multi2 * z + z_offset2
                        fmap[i, j, k, 0] = z

                    if n_pts < z_pix:
                        fmap[i, j, n_pts:, 0] = np.nan

        for j in range(y_pix):
            for i in range(x_pix):
                with zf.open(f"index/{i+j*x_pix}/segments/0/channels/vDeflection.dat") as f:
                    b = f.read()
                    fh = io.BytesIO(b)
                    n_pts = len(b)//4

                    for k in range(n_pts):
                        raw = struct.unpack(">i", fh.read(4))[0]
                        f_val = f_multi1 * raw + f_offset1
                        f_val = f_multi2 * f_val + f_offset2
                        f_val = f_multi3 * f_val + f_offset3
                        fmap[i, j, k, 1] = f_val

                    if n_pts < z_pix:
                        fmap[i, j, n_pts:, 1] = np.nan

                x = fmap[i, j, :, 0]
                y = fmap[i, j, :, 1]

                valid = np.isfinite(x) & np.isfinite(y)
                x = x[valid]
                y = y[valid]

                if len(x) < 2:
                    y_new = np.zeros(N_Z_PTS)
                else:
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]

                    x_unique, unique_idx = np.unique(x, return_index=True)
                    y_unique = y[unique_idx]

                    if len(x_unique) < 2:
                        y_new = np.zeros(N_Z_PTS)
                    else:
                        f_interp = interpolate.interp1d(
                            x_unique,
                            y_unique,
                            kind="linear",
                            fill_value="extrapolate"
                        )

                        x_new = np.linspace(
                            np.nanmin(x_unique),
                            np.nanmax(x_unique),
                            N_Z_PTS
                        )

                        y_new = f_interp(x_new)

                fcs.append(y_new)

    # ====================================================
    # PREPROCESS EXACTLY LIKE TRAINING
    # ====================================================

    X = np.array(fcs)   # shape = (N, 256)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-12

    X_norm = (X - mean) / std

    X_deriv = np.diff(
        X_norm,
        axis=1,
        prepend=X_norm[:, :1]
    )

    X_second = np.diff(
        X_deriv,
        axis=1,
        prepend=0
    )

    X_final = np.stack(
        [X_norm, X_deriv, X_second],
        axis=1
    )

    X_tensor = torch.tensor(X_final, dtype=torch.float32)

    print("Input tensor shape:", X_tensor.shape)

    # ====================================================
    # LOAD MODEL AND PREDICT
    # ====================================================

    print("Loading PyTorch model: best_resnet_final_4.pth")
    model = torch.jit.load("resnet1d_traced.pt")

    model.eval()

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)
        y_pred = torch.argmax(probs, dim=1).cpu().numpy()

    print("Prediction counts:", np.bincount(y_pred, minlength=3))
    print("First 10 probabilities:")
    print(probs[:10].cpu().numpy())

    # ====================================================
    # SAVE RESULT TSV
    # ====================================================

    with open(OUT_FILE, "w") as f:
        for j in range(y_pix):
            for i in range(x_pix):
                f.write(str(y_pred[j*x_pix+i]))

                if i != x_pix - 1:
                    f.write("\t")
                elif j != y_pix - 1:
                    f.write("\n")

    print("Classification result saved to", OUT_FILE)

    # ====================================================
    # PLOT HEATMAP
    # ====================================================

    heatmap = y_pred.reshape(y_pix, x_pix)

    cmap = ListedColormap([
        "#4daf4a",
        "#377eb8",
        "#e41a1c"
    ])

    norm = BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5],
        cmap.N
    )

    plt.figure(dpi=300)

    im = plt.imshow(
        heatmap,
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
        extent=(-0.5, x_pix-0.5, -0.5, y_pix-0.5)
    )

    cbar = plt.colorbar(im, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels([
        "Class 0",
        "Class 1",
        "Class 2"
    ])

    plt.title("Force-Curve Classification")
    plt.xlabel("X pixel")
    plt.ylabel("Y pixel")

    ax = plt.gca()
    ax.set_xticks(np.arange(0, x_pix, 1))
    ax.set_yticks(np.arange(0, y_pix, 1))
    ax.set_xticks(np.arange(-0.5, x_pix, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, y_pix, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.4)
    ax.tick_params(which="both", length=0)

    plt.tight_layout()
    plt.savefig(OUT_FILE + ".png", dpi=300)
    plt.savefig(OUT_FILE + ".eps")
    plt.close()

    print("Heatmap saved to:", OUT_FILE + ".png", "and", OUT_FILE + ".eps")