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

    for num in range(len(df)):
        # make case
        case = Case(
            diameter_cell=df.loc[num, "diameter_cell"],
            diameter_product=df.loc[num, "diameter_product"],
            thickness_rib=df.loc[num, "thickness_rib"],
            thickness_slit=df.loc[num, "thickness_slit"],
            thickness_product=df.loc[num, "thickness_product"],
            ratio_slit=df.loc[num, "ratio_slit"],
            mode_cell=df.loc[num, "mode_cell"],
            mode_slit=df.loc[num, "mode_slit"],
            thickness_bot=df.loc[num, "thickness_bot"],
            thickness_mid=df.loc[num, "thickness_mid"],
            thickness_top=df.loc[num, "thickness_top"],
            length_product=df.loc[num, "length_product"],
            ocell_rib=df.loc[num, "ocell_rib"],
            ocell_hight=df.loc[num, "ocell_hight"],
            ocell_x1=df.loc[num, "ocell_x1"],
            ocell_y1=df.loc[num, "ocell_y1"],
        )

        # set base slit positions
        positions_bslit = set_pos_slit(case.p_slit, case.d_prod, case.m_slit)

        # set base cell positions
        positions_bicell, positions_bocell = set_pos_cell(
            positions_bslit,
            case.p_cell,
            0.6 * case.d_prod,
            case.y_cell_slit,
            case.y_cell_cell,
            case.r_slit,
            case.m_cell,
        )

        # select slit
        positions_slit = select_slit(positions_bslit, case.lim_slit)

        # select cell
        lim_icell = 0.5 * case.d_prod - case.t_prod - 0.5 * case.d_cell
        lim_ocell = 0.5 * case.d_prod - case.t_prod - 0.5 * (case.p_cell - case.o_rib)
        positions_icell = select_cell(positions_bicell, lim_icell)
        positions_ocell = select_cell(positions_bocell, lim_ocell)

        # calc value
        effective_diameter_cell = case.d_cell - 2.0 * (case.t_bot + case.t_mid + case.t_top)
        num_icell = len(positions_icell)
        num_ocell = len(positions_ocell)
        df.loc[num, "num_icell"] = num_icell
        df.loc[num, "num_ocell"] = num_ocell
        df.loc[num, "num_slit"] = len(positions_slit)
        df.loc[num, "membrane_area"] = np.pi * effective_diameter_cell * case.length * num_icell

        icell_area = 0.25 * np.pi * (effective_diameter_cell ** 2.0) * num_icell
        ocell_area = (case.o_hight * (case.p_cell - case.o_rib) - 2.0 * case.o_x1 * case.o_y1) * num_ocell
        prod_area = 0.25 * np.pi * (case.d_prod ** 2.0)
        df.loc[num, "icell_area"] = icell_area
        df.loc[num, "ocell_area"] = ocell_area
        df.loc[num, "icell_area_prod_area"] = icell_area / prod_area
        df.loc[num, "ocell_area_prod_area"] = ocell_area / prod_area

        # data output
        cnt_str = f"index{num:0>4}"
        np.savetxt(dir_pos / f"{cnt_str}_icell.csv", np.round(positions_icell, decimals=15), fmt="%.14e", delimiter=",")
        np.savetxt(dir_pos / f"{cnt_str}_ocell.csv", np.round(positions_ocell, decimals=15), fmt="%.14e", delimiter=",")
        np.savetxt(dir_pos / f"{cnt_str}_slit.csv", np.round(positions_slit, decimals=15), fmt="%.14e", delimiter=",")

        parameters = df.iloc[num, :].T.to_dict()
        draw(parameters, positions_icell, positions_ocell, positions_slit, dir_path / cnt_str)
        df.to_csv(dir_path / "data.csv")

        del (
            case,
            positions_bslit,
            positions_slit,
            positions_bicell,
            positions_bocell,
            positions_icell,
            positions_ocell,
        )

    return None


class Case:
    def __init__(
        self,
        diameter_cell: float,
        diameter_product: float,
        thickness_rib: float,
        thickness_slit: float,
        thickness_product: float,
        thickness_bot: float,
        thickness_mid: float,
        thickness_top: float,
        length_product: float,
        ratio_slit: int,
        mode_cell: str,
        mode_slit: str,
        ocell_rib: float,
        ocell_hight: float,
        ocell_x1: float,
        ocell_y1: float,
    ):
        self.d_cell = diameter_cell
        self.d_prod = diameter_product
        self.t_rib = thickness_rib
        self.t_slit = thickness_slit
        self.t_prod = thickness_product
        self.t_bot = thickness_bot
        self.t_mid = thickness_mid
        self.t_top = thickness_top
        self.length = length_product
        self.r_slit = ratio_slit
        self.m_cell = mode_cell
        self.m_slit = mode_slit
        self.o_rib = ocell_rib
        self.o_hight = ocell_hight
        self.o_x1 = ocell_x1
        self.o_y1 = ocell_y1

        self.p_cell = self.d_cell + self.t_rib
        self.y_cell_cell = self.p_cell * np.sin(np.pi / 3.0)
        self.p_slit = self.t_slit + (self.r_slit - 1) * self.y_cell_cell + self.d_cell + 2.0 * self.t_rib
        self.lim_slit = 0.5 * self.d_prod - self.t_prod - self.d_cell - self.t_rib - 0.5 * self.t_slit
        self.y_cell_slit = 0.5 * self.t_slit + self.t_rib + 0.5 * self.d_cell


def set_parameters():
    # unit : mm
    # incell parameters
    diameter_cell = [10.27]
    diameter_product = [180.0]
    thickness_rib = [1.7]
    thickness_slit = [3.684]
    thickness_product = [5.0]
    thickness_bot = [250e-3]
    thickness_mid = [20e-3]
    thickness_top = [1e-3]
    length_product = [1000.0]
    ratio_slit = [2]
    mode_cell = ["A"]
    mode_slit = ["A"]
    # outcell parameters
    ocell_rib = [3.684 * 0.5]
    ocell_hight = [1.8]
    ocell_x1 = [0.8]
    ocell_y1 = [0.3]
    # calc value
    num_icell = [-1]
    num_ocell = [-1]
    num_slit = [-1]
    membrane_area = [-1.0]
    icell_area = [-1.0]
    ocell_area = [-1.0]
    icell_area_prod_area = [-1.0]
    ocell_area_prod_area = [-1.0]

    param_set = list(
        itertools.product(
            diameter_cell,
            diameter_product,
            thickness_rib,
            thickness_slit,
            thickness_product,
            thickness_bot,
            thickness_mid,
            thickness_top,
            length_product,
            ratio_slit,
            mode_cell,
            mode_slit,
            ocell_rib,
            ocell_hight,
            ocell_x1,
            ocell_y1,
            num_icell,
            num_ocell,
            num_slit,
            membrane_area,
            icell_area,
            ocell_area,
            icell_area_prod_area,
            ocell_area_prod_area,
        )
    )

    title = [
        "diameter_cell",
        "diameter_product",
        "thickness_rib",
        "thickness_slit",
        "thickness_product",
        "thickness_bot",
        "thickness_mid",
        "thickness_top",
        "length_product",
        "ratio_slit",
        "mode_cell",
        "mode_slit",
        "ocell_rib",
        "ocell_hight",
        "ocell_x1",
        "ocell_y1",
        "num_icell",
        "num_ocell",
        "num_slit",
        "membrane_area",
        "icell_area",
        "ocell_area",
        "icell_area_prod_area",
        "ocell_area_prod_area",
    ]

    return pd.DataFrame(param_set, columns=title)


def set_pos_slit(s_pitch: float, s_lim: float, s_mode: str):
    result = []
    i = 0
    while True:
        if s_mode == "A":
            aaa = s_pitch * i

        else:
            aaa = s_pitch * (i + 0.5)

        if aaa >= s_lim:
            break

        result.append(aaa)
        i += 1

    bbb = [xxx for xxx in result if xxx > 0.0]

    [result.append(-_) for _ in bbb]

    return np.array(result)


def set_pos_cell(
    pos_slit: list, c_pitch: float, c_lim: float, cs_pitch: float, cc_pitch: float, r_slit: int, c_mode: str
):
    icell = []
    ocell = []

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

    return np.array(icell), np.array(ocell)


def select_slit(bslit: list, s_lim: float):
    return np.array([xxx for xxx in bslit if abs(xxx) < s_lim])


def select_cell(bcell: list, c_lim: float):
    result = []
    for xy in bcell:
        radius = np.sqrt(xy[0] ** 2.0 + xy[1] ** 2.0)
        if radius < c_lim:
            result.append([xy[0], xy[1]])

    return np.array(result)


def draw(param: dict, pos_icell: list, pos_ocell: list, pos_slit: list, fpath: str) -> None:
    def draw_circle(ax, radius: float, center: tuple = (0.0, 0.0), linestyle: str = "solid") -> None:
        theta = np.linspace(0.0, 2.0 * np.pi, 360)

        xxx = center[0] + radius * np.cos(theta)
        yyy = center[1] + radius * np.sin(theta)

        ax.plot(xxx, yyy, color="black", linestyle=linestyle)

        return None

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
                x_t = [-d_prod, d_prod]
                x_b = [-d_prod, d_prod]

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
    fig = plt.figure(figsize=(8, 5))
    spec = GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])

    ax = fig.add_subplot(spec[0])
    ax.grid(linewidth=0.2)
    ax.set_axisbelow(True)

    aaa = 0.5 * param["diameter_product"] + 10.0
    ax.set_xlim(-aaa, aaa)
    ax.set_ylim(-aaa, aaa)

    # draw product
    draw_circle(ax, radius=0.5 * param["diameter_product"])
    draw_circle(ax, radius=(0.5 * param["diameter_product"] - param["thickness_product"]), linestyle="dashed")

    # draw slit
    draw_slit(ax, pos_slit, param["thickness_slit"], param["diameter_product"])

    # draw icell
    for _ in pos_icell:
        r_cell = 0.5 * param["diameter_cell"] - (
            param["thickness_bot"] + param["thickness_mid"] + param["thickness_top"]
        )
        ax.add_patch(patch.Circle(xy=_, radius=r_cell))

    # draw ocell
    for xy in pos_ocell:
        draw_ocell(
            ax,
            param["diameter_cell"] + param["thickness_rib"],
            param["ocell_rib"],
            param["ocell_hight"],
            param["ocell_x1"],
            param["ocell_y1"],
            (xy[0], xy[1]),
        )

    # draw table
    temp_str = []
    table_label = []
    for k, v in param.items():
        if k in ["membrane_area", "icell_area", "ocell_area"]:
            temp_str.append([f"{v:.3e} [mm2]"])
            table_label.append(k)

        elif k in ["icell_area_prod_area", "ocell_area_prod_area"]:
            temp_str.append([f"{v*100.0:.3g} [%]"])
            table_label.append(k)

        elif k in ["ratio_slit", "num_icell", "num_ocell", "num_slit"]:
            temp_str.append([f"{v} [-]"])
            table_label.append(k)

        elif k in ["mode_cell", "mode_slit"]:
            temp_str.append([str(v)])
            table_label.append(k)

        else:
            temp_str.append([f"{v} [mm]"])
            table_label.append(k)

    ay = fig.add_subplot(spec[1])
    ay.axis("off")
    ay.table(cellText=temp_str, rowLabels=table_label, loc="center", bbox=[0.7, 0, 1, 1])
    # bbox = (xmin, ymin, width, height)

    # save fig
    fig.savefig(fpath, facecolor="w", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return None


def dump_json(data: dict, fname: str = "test_json", sw_sort: bool = False):
    with open(fname, "wt") as f:
        json.dump(data, f, sort_keys=sw_sort, index=4)

    return None


if __name__ == "__main__":
    main()
    print("****** Done ******")


# %%
