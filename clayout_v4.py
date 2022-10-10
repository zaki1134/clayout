# %%
from dataclasses import dataclass
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from itertools import product, zip_longest
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
    inp = set_parameters()
    out = pd.DataFrame()

    for num in range(len(inp)):
        cnt_str = f"index{num:0>4}"
        case = Position(**inp.loc[num, :])
        params = inp.loc[num, :].to_dict()

        # validate parameters
        ecode = case.validation()
        if not len(ecode) == 0:
            with open(dir_path / "error.txt", "at") as f:
                temp = f"{num}, {[_ for _ in ecode]}"
                f.write(temp)
                f.write("\n")
            continue

        # set base slit positions
        case.set_base_slit()

        # set base cell positions
        case.set_base_cell()
        case.distribute_cell()
        case.offset_xy()

        # select slit
        case.select_slit()

        # select cell
        case.select_icell()
        case.select_ocell()

        # calculate values
        res = case.calc_values()
        res.index = [num]
        out = pd.concat([out, res])

        # output positions
        case.output_positions(dir_pos / cnt_str, params)

        # output png
        case.output_draw(dir_path / cnt_str, out.loc[num, :].to_dict())

        del case, res

    # output parameters
    out = pd.concat([inp, out], axis=1)
    out.to_csv(dir_path / "data.csv")


def set_parameters():
    # unit : mm
    param = {
        # product
        "dia_prod": [30.0],
        "thk_prod": [1.0],
        "thk_wall": [0.5],
        "thk_slit": [1.0],
        "thk_c2s": [1.0],
        "ln_prod": [1000.0],
        "ln_slit": [30.0],
        "ln_seal": [5.0],
        "ln_glass_seal": [15.0],
        "ratio_slit": [3],
        "mode_cell": [True],
        "mode_slit": [True],
        # membrane
        "thk_bot": [250e-3],
        "thk_mid": [20e-3],
        "thk_top": [1e-3],
        # icell
        "shape_icell": [1],
        "dia_icell": [1.5],
        # ocell
        "shape_ocell": [1],
        "thk_wall_ocell": [0.5],
        "hgt_ocell": [1.0],
        "x1_ocell": [0.2],
        "y1_ocell": [0.2],
    }

    _ = list(product(*param.values()))

    return pd.DataFrame(_, columns=[k for k in param.keys()])


@dataclass
class Params:
    # product
    dia_prod: float
    thk_prod: float
    thk_wall: float
    ln_prod: float
    ln_seal: float
    ln_glass_seal: float
    thk_c2s: float
    ratio_slit: int
    mode_cell: bool
    mode_slit: bool
    # membrane
    thk_bot: float
    thk_mid: float
    thk_top: float
    # slit
    thk_slit: float
    ln_slit: float
    # incell
    shape_icell: int
    dia_icell: float
    # outcell
    shape_ocell: int
    thk_wall_ocell: float
    hgt_ocell: float
    x1_ocell: float
    y1_ocell: float
    # supported_shape
    supported_shape_in = {1: "CIR", 2: "HEX"}
    supported_shape_out = {1: "OCT", 2: "CIR"}

    def pitch_x(self):
        if self.shape_icell == 1:
            result = self.dia_icell + self.thk_wall
        elif self.shape_icell == 2:
            pass

        return result

    def pitch_y(self):
        if self.shape_icell == 1:
            result = self.pitch_x() * np.sin(np.pi / 3.0)
        elif self.shape_icell == 2:
            pass

        return result

    def pitch_slit(self):
        if self.shape_icell == 1:
            _ = self.thk_slit + self.dia_icell + 2.0 * self.thk_c2s
            result = _ + (self.ratio_slit - 1) * self.pitch_y()
        elif self.shape_icell == 2:
            pass

        return result

    def lim_slit(self, num: int = 1):
        if self.shape_icell == 1:
            _ = 0.5 * (self.dia_prod - self.thk_slit)
            result = _ - self.thk_prod - num * self.dia_icell - self.thk_c2s
        elif self.shape_icell == 2:
            pass

        return result

    def lim_icell(self):
        if self.shape_icell == 1:
            result = 0.5 * self.dia_prod - self.thk_prod - 0.5 * self.dia_icell
        elif self.shape_icell == 2:
            pass

        return result

    def lim_ocell(self):
        if self.shape_ocell == 1:
            result = 0.5 * self.dia_prod - self.thk_prod - 0.5 * (self.pitch_x() - self.thk_wall_ocell)
        elif self.shape_ocell == 2:
            pass

        return result


@dataclass
class Position(Params):
    base_slit = np.empty([0, 2])
    base_cell = np.empty([0, 2])
    full_icell = np.empty([0, 2])
    full_ocell = np.empty([0, 2])
    pos_slit = np.empty([0, 2])
    pos_icell = np.empty([0, 2])
    pos_ocell = np.empty([0, 2])

    def validation(self):
        code = []

        # check 1
        if self.shape_icell not in self.supported_shape_in.keys():
            code.append(1)

        # check 2
        if self.shape_ocell not in self.supported_shape_out.keys():
            code.append(2)

        # check 3
        if not self.dia_icell - 2 * (self.thk_top + self.thk_mid + self.thk_bot) >= 0:
            code.append(3)

        # check 4
        if not self.thk_slit >= self.hgt_ocell:
            code.append(4)

        # check 5
        if not 0.5 * self.hgt_ocell >= self.y1_ocell:
            code.append(5)

        # check 6
        if not 0.5 * (self.pitch_x() - self.thk_wall_ocell) >= self.x1_ocell:
            code.append(6)

        # check 7
        if not (self.dia_prod - self.thk_prod) >= self.dia_icell:
            code.append(7)

        # check 8
        if not 0.5 * self.ln_prod >= (self.ln_glass_seal + self.ln_slit):
            code.append(8)

        return code

    def set_base_slit(self, num: int = 300):
        # uniform slit length
        y1 = []
        plus = [self.pitch_slit() * i for i in range(num) if self.dia_prod >= self.pitch_slit() * i]
        y1[len(y1) : len(y1)] = plus
        y1[len(y1) : len(y1)] = [-x for x in plus if x > 0.0]

        y1.sort()
        _ = [[y, z] for y, z in zip_longest(y1, [self.ln_slit], fillvalue=self.ln_slit)]
        self.base_slit = np.array(_)

    def set_base_cell(self, num: int = 300):
        # incell & outcell base positions
        x1 = []
        plus = [self.pitch_x() * i for i in range(num)]
        x1[len(x1) : len(x1)] = plus
        x1[len(x1) : len(x1)] = [-x for x in plus if x > 0.0]

        x2 = [x + (0.5 * self.pitch_x()) for x in x1]

        o2i_y = self.thk_c2s + 0.5 * (self.thk_slit + self.dia_icell)
        y1 = [0.0, o2i_y]
        y1[len(y1) : len(y1)] = [o2i_y + self.pitch_y() * i for i in range(1, self.ratio_slit)]

        _ = []
        for i in range(len(y1)):
            if i % 2 == 0:
                [_.append([x, y1[i]]) for x in x1]
            else:
                [_.append([x, y1[i]]) for x in x2]
        self.base_cell = np.array(_)

    def distribute_cell(self):
        # full_icell & full_ocell positions
        yyy = self.base_slit[:, 0].tolist()
        if (self.ratio_slit + 1) % 2 == 0:
            xxx = [0.0] * len(yyy)
        else:
            xxx = [0.5 * self.pitch_x() if i % 2 != 0 else 0.0 for i in range(len(yyy))]

        oc = []
        ic = []
        for x1, y1 in zip(xxx, yyy):
            _ = [[x + x1, y + y1] for x, y in self.base_cell.tolist() if y == 0.0]
            oc[len(oc) : len(oc)] = _
            _ = [[x + x1, y + y1] for x, y in self.base_cell.tolist() if y != 0.0]
            ic[len(ic) : len(ic)] = _

        self.full_icell = np.array(ic)
        self.full_ocell = np.array(oc)

    def offset_xy(self):
        # full_icell & full_ocell XYdir_offset
        if self.mode_cell:
            pass
        else:
            x1 = 0.5 * self.pitch_x()
            ic = [[x + x1, y] for x, y in self.full_icell.tolist()]
            oc = [[x + x1, y] for x, y in self.full_ocell.tolist()]
            self.full_icell = np.array(ic)
            self.full_ocell = np.array(oc)

        if self.mode_slit:
            pass
        else:
            y1 = 0.5 * self.pitch_slit()
            ic = [[x, y + y1] for x, y in self.full_icell.tolist()]
            oc = [[x, y + y1] for x, y in self.full_ocell.tolist()]
            sl = [[y + y1, z] for y, z in self.base_slit.tolist()]
            self.full_icell = np.array(ic)
            self.full_ocell = np.array(oc)
            self.base_slit = np.array(sl)

    def select_slit(self):
        _ = [[y, z] for y, z in self.base_slit.tolist() if self.lim_slit() >= abs(y)]
        self.pos_slit = np.array(_)

    def select_icell(self):
        _ = [[x, y] for x, y in self.full_icell.tolist() if self.lim_icell() >= np.sqrt(x**2 + y**2)]
        self.pos_icell = np.array(_)

    def select_ocell(self):
        _ = [[x, y] for x, y in self.full_ocell.tolist() if self.lim_ocell() >= np.sqrt(x**2 + y**2)]
        _ = [[x, y] for x, y in _ if self.lim_slit() >= abs(y)]
        self.pos_ocell = np.array(_)

    def calc_values(self):
        # unit : mm
        result = {}
        area_prod = 0.25 * np.pi * (self.dia_prod**2.0)
        vol_prod = area_prod * self.ln_prod
        ln_ocell = self.ln_prod - 2.0 * (self.ln_seal + self.ln_slit)
        if self.shape_icell == 1:
            eff_dia_icell = self.dia_icell - 2.0 * (self.thk_bot + self.thk_mid + self.thk_top)
            area_icell = 0.25 * np.pi * (eff_dia_icell**2.0)
            eff_peri = np.pi * eff_dia_icell
        elif self.shape_icell == 2:
            pass

        if self.shape_ocell == 1:
            area_sq = self.hgt_ocell * (self.pitch_x() - self.thk_wall_ocell)
            area_corner = 2.0 * (self.x1_ocell * self.y1_ocell)
            area_ocell = area_sq - area_corner
        elif self.shape_icell == 1:
            pass

        result["N_icell"] = self.pos_icell.shape[0]
        result["N_ocell"] = self.pos_ocell.shape[0]
        result["N_slit"] = self.pos_slit.shape[0]
        result["A_membrane"] = eff_peri * self.ln_prod * self.pos_icell.shape[0]
        result["A_icell"] = area_icell * self.pos_icell.shape[0]
        result["A_ocell"] = area_ocell * self.pos_ocell.shape[0]
        result["R(A_icell/A_prod)"] = result["A_icell"] / area_prod
        result["R(A_ocell/A_prod)"] = result["A_ocell"] / area_prod
        result["V_icell"] = result["A_icell"] * self.ln_prod
        result["V_ocell"] = result["A_ocell"] * ln_ocell
        result["R(V_icell/V_prod)"] = result["V_icell"] / vol_prod
        result["R(V_ocell/V_prod)"] = result["V_ocell"] / vol_prod

        return pd.DataFrame.from_dict(result, orient="index").T

    def output_draw(self, fpath: str, res: dict, dbg: bool = False):
        def product(radius: float, style: str = "solid", num: int = 360):
            theta = np.linspace(0.0, 2.0 * np.pi, num)
            xxx = 0.0 + radius * np.cos(theta)
            yyy = 0.0 + radius * np.sin(theta)

            ax.plot(xxx, yyy, color="black", linestyle=style)

        def slit():
            if not dbg:
                yyy = [y for y, _ in self.pos_slit.tolist()]
            else:
                yyy = [y for y, _ in self.base_slit.tolist()]

            y_t = [yyy + 0.5 * self.thk_slit] * 2
            y_b = [yyy - 0.5 * self.thk_slit] * 2

            if not dbg:
                _ = (0.5 * self.dia_prod) ** 2.0
                x_t = [-np.sqrt(_ - y_t[0] ** 2.0), np.sqrt(_ - y_t[1] ** 2.0)]
                x_b = [-np.sqrt(_ - y_b[0] ** 2.0), np.sqrt(_ - y_b[1] ** 2.0)]
            else:
                x_t = [-self.dia_prod, self.dia_prod]
                x_b = [-self.dia_prod, self.dia_prod]

            ax.plot(x_t, y_t, color="blue", linewidth=0.6, linestyle="dashed")
            ax.plot(x_b, y_b, color="blue", linewidth=0.6, linestyle="dashed")

        def icell():
            if self.shape_icell == 1:
                if not dbg:
                    r_cell = 0.5 * self.dia_icell - (self.thk_bot + self.thk_mid + self.thk_top)
                    color = "tab:blue"
                else:
                    r_cell = 0.5 * self.dia_icell
                    color = "gray"

                [ax.add_patch(patches.Circle(xy=tuple(_), radius=r_cell, fc=color)) for _ in self.pos_icell.tolist()]
            elif self.shape_icell == 2:
                pass
            else:
                pass

        def ocell():
            ref = []
            if self.shape_ocell == 1:
                dx = 0.5 * (self.pitch_x() - self.thk_wall_ocell)
                dy = 0.5 * self.hgt_ocell
                ref.append([dx, dy - self.y1_ocell])
                ref.append([dx - self.x1_ocell, dy])
                ref.append([-ref[1][0], ref[1][1]])
                ref.append([-ref[0][0], ref[0][1]])
                ref.append([ref[3][0], -ref[3][1]])
                ref.append([ref[2][0], -ref[2][1]])
                ref.append([ref[1][0], -ref[1][1]])
                ref.append([ref[0][0], -ref[0][1]])

                _ = np.array(ref)
                ref = np.insert(_, [2], 1.0, axis=1)
                for xxx, yyy in self.pos_ocell.tolist():
                    tra = np.array([[1, 0, xxx], [0, 1, yyy], [0, 0, 1]])
                    _ = np.array([tra @ xy for xy in ref])
                    res = np.delete(_, [2], axis=1)

                    _ = plt.Polygon(res, fc="green")
                    ax.add_patch(_)

            elif self.shape_ocell == 2:
                pass
            else:
                pass

        def table():
            exc = [
                "base_slit",
                "base_cell",
                "full_icell",
                "full_ocell",
                "pos_slit",
                "pos_icell",
                "pos_ocell",
            ]
            inp = dict([[k, v] for k, v in self.__dict__.items() if k not in exc])
            inp.update(res)

            exc = [
                "R(A_icell/A_prod)",
                "R(A_ocell/A_prod)",
                "R(V_icell/V_prod)",
                "R(V_ocell/V_prod)",
                "V_icell",
                "V_ocell",
            ]
            f00 = ["A_membrane", "A_icell", "A_ocell"]
            f01 = []
            f02 = []
            f03 = ["ratio_slit", "N_icell", "N_ocell", "N_slit"]
            f04 = ["mode_cell", "mode_slit"]
            f05 = ["shape_icell", "shape_ocell"]

            key = []
            val = []
            for k, v in inp.items():
                if k in f00:
                    key.append(k)
                    val.append([f"{v:.3e} [mm2]"])
                elif k in f01:
                    key.append(k)
                    val.append([f"{v:.3e} [mm3]"])
                elif k in f02:
                    key.append(k)
                    val.append([f"{v*100.0:.3g} [%]"])
                elif k in f03:
                    key.append(k)
                    val.append([f"{v:.0f}"])
                elif k in f04:
                    key.append(k)
                    if v:
                        val.append(["A"])
                    else:
                        val.append(["B"])
                elif k in f05:
                    key.append(k)
                    val.append([str(v)])
                elif k in exc:
                    pass
                else:
                    key.append(k)
                    val.append([f"{v} [mm]"])

            ay = fig.add_subplot(spec[1])
            ay.axis("off")
            # bbox = (xmin, ymin, width, height)
            tab = ay.table(cellText=val, rowLabels=key, loc="left", bbox=[0.2, 0.0, 1.0, 1.0])
            tab.set_fontsize(8)

        # make fig, ax
        fig = plt.figure(figsize=(10, 6), dpi=300)
        spec = GridSpec(ncols=2, nrows=1, width_ratios=[2, 1])

        ax = fig.add_subplot(spec[0])
        ax.grid(linewidth=0.2)
        ax.set_axisbelow(True)

        xylim = 0.52 * self.dia_prod
        ax.set_xlim(-xylim, xylim)
        ax.set_ylim(-xylim, xylim)

        # draw product
        product(radius=0.5 * self.dia_prod)
        product(radius=(0.5 * self.dia_prod - self.thk_prod), style="dashed")

        # draw slit
        slit()

        # draw cell
        icell()
        ocell()

        # draw table
        table()

        # save fig
        fig.savefig(fpath, facecolor="w", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def output_positions(self, fpath, params):
        xxx = fpath.with_name(str(fpath.name) + "_pos_icell.csv")
        np.savetxt(xxx, self.pos_icell, fmt="%.14e", delimiter=",")
        xxx = fpath.with_name(str(fpath.name) + "_pos_ocell.csv")
        np.savetxt(xxx, self.pos_ocell, fmt="%.14e", delimiter=",")
        xxx = fpath.with_name(str(fpath.name) + "_pos_slit.csv")
        np.savetxt(xxx, self.pos_slit, fmt="%.14e", delimiter=",")

        xxx = fpath.with_name(str(fpath.name) + "_parameters.json")
        with open(xxx, "w") as f:
            json.dump(params, f, sort_keys=False, indent=4)


if __name__ == "__main__":
    main()
    print("****** Done ******")


# %%
