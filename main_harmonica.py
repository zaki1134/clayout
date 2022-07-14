# %%
import numpy as np
import json

import itertools
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib.gridspec import GridSpec

from pathlib import Path
from datetime import datetime


def main():
    # make dir
    dir_path = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    dir_pos = dir_path / Path("positions_data")
    if not dir_path.exists():
        dir_path.mkdir()
        dir_pos.mkdir()

    # set parameters
    df = set_parameters()
    df.to_csv(dir_path / "data.csv")

    for num in range(len(df)):
        # temporary parameters
        temp_param = {}
        temp_param["pitch_cell"] = df.loc[num, "diameter_icell"] + df.loc[num, "thickness_rib"]
        temp_param["y_cell_cell"] = temp_param["pitch_cell"] * np.sin(np.pi / 3.0)
        temp_param["pitch_slit"] = (
            df.loc[num, "thickness_slit"]
            + (df.loc[num, "ratio_slit"] - 1) * temp_param["y_cell_cell"]
            + df.loc[num, "diameter_icell"]
            + 2.0 * df.loc[num, "thickness_rib"]
        )
        temp_param["y_cell_slit"] = (
            0.5 * df.loc[num, "thickness_slit"] + df.loc[num, "thickness_rib"] + 0.5 * df.loc[num, "diameter_icell"]
        )
        # set slit positions
        positions_slit = [0.0, temp_param["pitch_slit"]]

        # set base cell positions
        positions_bicell, positions_bocell = set_pos_cell(
            positions_slit,
            temp_param["pitch_cell"],
            0.6 * df.loc[num, "diameter_prod"],
            1.2 * temp_param["pitch_slit"],
            temp_param["y_cell_slit"],
            temp_param["y_cell_cell"],
            df.loc[num, "ratio_slit"],
        )

        # select cell
        lim_icell = (
            0.5 * df.loc[num, "diameter_prod"] - df.loc[num, "thickness_prod"] - 0.5 * df.loc[num, "diameter_icell"]
        )
        lim_ocell = (
            0.5 * df.loc[num, "diameter_prod"]
            - df.loc[num, "thickness_prod"]
            - 0.5 * (temp_param["pitch_cell"] - df.loc[num, "thickness_rib_ocell"])
        )
        positions_icell = select_cell(positions_bicell, lim_icell, temp_param["pitch_slit"] + temp_param["y_cell_slit"])
        positions_ocell = select_cell(positions_bocell, lim_ocell, temp_param["pitch_slit"] + temp_param["y_cell_slit"])

        # data output
        cnt_str = f"index{num:0>4}"
        np.savetxt(dir_pos / f"{cnt_str}_icell.csv", np.round(positions_icell, decimals=15), fmt="%.14e", delimiter=",")
        np.savetxt(dir_pos / f"{cnt_str}_ocell.csv", np.round(positions_ocell, decimals=15), fmt="%.14e", delimiter=",")
        np.savetxt(dir_pos / f"{cnt_str}_slit.csv", np.round(positions_slit, decimals=15), fmt="%.14e", delimiter=",")

        parameters = df.iloc[num, :].T.to_dict()
        dump_json(parameters, dir_pos / f"{cnt_str}_param.json")
        draw(temp_param, parameters, positions_icell, positions_ocell, positions_slit, dir_path / cnt_str)

        del (
            temp_param,
            positions_slit,
            positions_bicell,
            positions_bocell,
            positions_icell,
            positions_ocell,
        )

    return None


def set_parameters():
    # unit : mm
    param = {
        # icell parameters
        "diameter_icell": [4.0],
        "diameter_prod": [30.0],
        "thickness_rib": [0.5],
        "thickness_slit": [1.0],
        "thickness_prod": [1.0],
        "thickness_bot": [250e-3],
        "thickness_mid": [20e-3],
        "thickness_top": [1e-3],
        "length_prod": [1000.0],
        "length_slit": [30.0],
        "length_seal": [40.0],
        "ratio_slit": [2, 3],
        # ocell parameters
        "thickness_rib_ocell": [0.5],
        "hight_ocell": [1.0],
        "width_x1_ocell": [0.0],
        "hight_y1_ocell": [0.0],
    }

    param_set = list(itertools.product(*param.values()))

    return pd.DataFrame(param_set, columns=[k for k in param.keys()])


def set_pos_cell(
    pos_slit: list,
    c_pitch: float,
    c_lim: float,
    s_lim: float,
    cs_pitch: float,
    cc_pitch: float,
    r_slit: int,
    c_mode: str = "A",
):
    icell = []
    ocell = []
    filter_ocell = []

    # x-coord preset
    cnt_x = int(np.ceil(c_lim / c_pitch))
    if c_mode == "A":
        x_enum = [c_pitch * i for i in range(cnt_x)]
        x_enum[len(x_enum) : len(x_enum)] = [c_pitch * i for i in range(-1, -cnt_x, -1)]
        x_onum = [c_pitch * (i + 0.5) for i in range(cnt_x)]
        x_onum[len(x_onum) : len(x_onum)] = [c_pitch * (i + 0.5) for i in range(-1, -cnt_x, -1)]

    else:
        x_enum = [c_pitch * (i + 0.5) for i in range(cnt_x)]
        x_enum[len(x_enum) : len(x_enum)] = [c_pitch * (i + 0.5) for i in range(-1, -cnt_x, -1)]
        x_onum = [c_pitch * i for i in range(cnt_x)]
        x_onum[len(x_onum) : len(x_onum)] = [c_pitch * i for i in range(-1, -cnt_x, -1)]

    # make icell, ocell
    cnt = 0
    for yyy in np.sort(pos_slit):
        for i in range(r_slit + 1):
            if i == 0:
                if cnt % 2 == 0:
                    [ocell.append([xxx, yyy]) for xxx in x_onum]

                else:
                    [ocell.append([xxx, yyy]) for xxx in x_enum]

            else:
                dy = yyy + cs_pitch + (i - 1) * cc_pitch
                if not i % 2 == 0:
                    if cnt % 2 == 0:
                        [icell.append([xxx, dy]) for xxx in x_onum]

                    else:
                        [icell.append([xxx, dy]) for xxx in x_enum]

                else:
                    if cnt % 2 == 0:
                        [icell.append([xxx, dy]) for xxx in x_onum]

                    else:
                        [icell.append([xxx, dy]) for xxx in x_enum]

            cnt += 1

    [filter_ocell.append([xy[0], xy[1]]) for xy in ocell if abs(xy[1]) < s_lim]

    return np.array(icell), np.array(filter_ocell)


def select_cell(bcell: list, x_lim: float, y_lim: float):
    result = []
    for xy in bcell:
        if -1e-15 < xy[0] < x_lim and xy[1] < y_lim:
            result.append([xy[0], xy[1]])

    return np.array(result)


def draw(temp_dir: dict, param: dict, pos_icell: list, pos_ocell: list, pos_slit: list, fpath: str) -> None:
    def draw_slit(ax, pos_slit: list, t_slit: float, d_prod: float, dbg: bool = True) -> None:
        for _ in pos_slit:
            y_t = [_ + 0.5 * t_slit] * 2
            y_b = [_ - 0.5 * t_slit] * 2

            if dbg:
                x_t = [
                    -np.sqrt((0.5 * d_prod) ** 2.0 - y_t[0] ** 2.0),
                    np.sqrt((0.5 * d_prod) ** 2.0 - y_t[1] ** 2.0),
                ]
                x_b = [
                    -np.sqrt((0.5 * d_prod) ** 2.0 - y_b[0] ** 2.0),
                    np.sqrt((0.5 * d_prod) ** 2.0 - y_b[1] ** 2.0),
                ]

            else:
                x_t = [0.0, 0.5 * d_prod]
                x_b = [0.0, 0.5 * d_prod]

            ax.plot(x_t, y_t, color="blue", linewidth=1, linestyle="dashed")
            ax.plot(x_b, y_b, color="blue", linewidth=1, linestyle="dashed")

        return None

    def draw_ocell(
        ax, c_pitch: float, o_rib: float, o_hight: float, x1: float, y1: float, center: tuple = (0.0, 0.0)
    ) -> None:
        point = []
        dx = c_pitch - o_rib - 2.0 * x1
        dy = o_hight - 2.0 * y1
        width = c_pitch - o_rib

        point.append([center[0] + 0.5 * (c_pitch - o_rib), center[1] + 0.5 * o_hight - y1])
        point.append([point[0][0] - x1, point[0][1] + y1])

        point.append([point[1][0] - dx, point[1][1]])
        point.append([point[0][0] - width, point[0][1]])

        point.append([point[0][0] - width, point[0][1] - dy])
        point.append([point[1][0] - dx, point[1][1] - o_hight])

        point.append([point[1][0], point[1][1] - o_hight])
        point.append([point[0][0], point[0][1] - dy])

        point.append([point[0][0], point[0][1]])

        xxx = [point[i][0] for i in range(len(point))]
        yyy = [point[i][1] for i in range(len(point))]

        ax.plot(xxx, yyy, color="green", linewidth=0.5, linestyle="dashed")

        return None

    # make fig, ax
    fig = plt.figure(figsize=(10, 6))
    spec = GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])

    ax = fig.add_subplot(spec[0])
    ax.grid(linewidth=0.2)
    ax.set_axisbelow(True)

    aaa = param["diameter_prod"]
    ax.set_xlim(-aaa, aaa)
    ax.set_ylim(-aaa, aaa)

    # draw product
    x1 = [0.0, 0.5 * param["diameter_prod"], 0.5 * param["diameter_prod"], 0.0, 0.0]
    y1 = [0.0, 0.0, temp_dir["pitch_slit"], temp_dir["pitch_slit"], 0.0]
    x2 = [
        0.5 * param["diameter_prod"] - param["thickness_prod"],
        0.5 * param["diameter_prod"] - param["thickness_prod"],
    ]
    y2 = [0.0, temp_dir["pitch_slit"]]
    ax.plot(x1, y1, color="black", linewidth=1, linestyle="solid")
    ax.plot(x2, y2, color="black", linewidth=1, linestyle="dashed")

    # draw slit
    draw_slit(ax, pos_slit, param["thickness_slit"], param["diameter_prod"], False)

    # draw icell
    for _ in pos_icell:
        r_cell = 0.5 * param["diameter_icell"] - (
            param["thickness_bot"] + param["thickness_mid"] + param["thickness_top"]
        )
        ax.add_patch(patch.Circle(xy=_, radius=r_cell))

    # draw ocell
    for xy in pos_ocell:
        draw_ocell(
            ax,
            param["diameter_icell"] + param["thickness_rib"],
            param["thickness_rib_ocell"],
            param["hight_ocell"],
            param["width_x1_ocell"],
            param["hight_y1_ocell"],
            (xy[0], xy[1]),
        )

    # draw table
    temp_str = []
    table_label = []
    for k, v in param.items():
        if k in ["A_membrane", "A_icell", "A_ocell"]:
            temp_str.append([f"{v:.3e} [mm2]"])
            table_label.append(k)

        elif k in ["V_icell", "V_ocell"]:
            temp_str.append([f"{v:.3e} [mm3]"])
            table_label.append(k)

        elif k in ["A_icell_A_prod", "A_ocell_A_prod", "V_icell_V_prod", "V_ocell_V_prod"]:
            temp_str.append([f"{v*100.0:.3g} [%]"])
            table_label.append(k)

        elif k in ["ratio_slit", "N_icell", "N_ocell", "N_slit"]:
            temp_str.append([f"{v} [-]"])
            table_label.append(k)

        else:
            temp_str.append([f"{v} [mm]"])
            table_label.append(k)

    ay = fig.add_subplot(spec[1])
    ay.axis("off")
    ay.table(cellText=temp_str, rowLabels=table_label, loc="center", bbox=[0.4, 0.0, 1.0, 1.0])
    # bbox = (xmin, ymin, width, height)

    # save fig
    fig.savefig(fpath, facecolor="w", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return None


def dump_json(data: dict, fname: str = "test_json", sw_sort: bool = False):
    with open(fname, "wt") as f:
        json.dump(data, f, sort_keys=sw_sort, indent=4)

    return None


if __name__ == "__main__":
    main()
    print("****** Done ******")


# %%