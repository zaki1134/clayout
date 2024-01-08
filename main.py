# %%
import sys
from pathlib import Path, WindowsPath
from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes._axes import Axes
from matplotlib.gridspec import GridSpec

from pprint import pprint as pp


def main():
    # check the number of arguments
    # if len(sys.argv) != 2:
    #     return None

    # make directory
    # path = make_dir()
    # if path is None:
    #     return None

    # read parameters
    # params = read_csv(sys.argv[1])
    params = read_csv("inp.csv")

    # main
    for row in params.itertuples():
        # base slit
        base_slit = CirOct.set_base_slit(
            row.dia_prod,
            row.pitch_slit,
        )

        # base cell
        base_incell, base_outcell = CirOct.set_base_cell(
            row.dia_prod,
            row.dia_incell,
            row.thk_slit,
            row.thk_outcell,
            row.thk_c2s,
            row.pitch_x,
            row.pitch_y,
            row.ratio_slit,
        )
        copied_incell = CirOct.copy_base_cell(
            row.pitch_x,
            row.ratio_slit,
            base_incell,
            base_slit,
        )
        copied_outcell = CirOct.copy_base_cell(
            row.pitch_x,
            row.ratio_slit,
            base_outcell,
            base_slit,
        )

        # offset cell, slit
        offset_incell, offset_outcell, offset_slit = CirOct.offset_xy(
            row.mode_cell,
            row.mode_slit,
            row.pitch_x,
            row.pitch_slit,
            copied_incell,
            copied_outcell,
            base_slit,
        )

        # select slit
        select_slit = CirOct.select_slit(
            row.lim_slit,
            offset_slit,
        )

        # select cell
        select_inell = CirOct.select_cell(
            row.lim_incell,
            offset_incell,
        )
        select_outell = CirOct.select_cell(
            row.lim_outcell,
            offset_outcell,
        )

        # draw
        Post.draw(
            row,
            select_inell,
            select_outell,
            select_slit,
        )


def make_dir() -> WindowsPath:
    """
    結果保存用フォルダの作成

    Returns:
        WindowsPath: 作成したフォルダパス
    """
    dir_path = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not dir_path.exists():
        dir_path.mkdir()
        res = dir_path
    else:
        res = None

    return res


def read_csv(path: str) -> pd.DataFrame:
    """
    コマンドライン引数のcsvファイルの読み込み

    Args:
        path (str): inp.csv

    Returns:
        pd.DataFrame: 全水準
    """
    if WindowsPath(path).exists():
        res = pd.read_csv(path)
    else:
        res = None

    return res


class CirOct:
    """
    incell : circle
    outcell : octagon

    """

    def set_base_slit(
        dia_prod: float,
        pitch_slit: float,
    ) -> np.ndarray:
        """
        slitの基準座標([Y])を算出

        Args:
            dia_prod (float): dia_prod
            pitch_slit (float): pitch_slit

        Returns:
            res (np.ndarray): ([Y])
        """

        num = np.int64(np.ceil(dia_prod / pitch_slit))
        res = [pitch_slit * i for i in range(num)]
        [res.append(pitch_slit * i) for i in range(-1, -num, -1)]
        res = np.array(res)
        res.sort()

        return res

    def set_base_cell(
        dia_prod: float,
        dia_incell: float,
        thk_slit: float,
        thk_outcell: float,
        thk_c2s: float,
        pitch_x: float,
        pitch_y: float,
        ratio_slit: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        incellとoutcellの基準座標([X, Y])を算出

        Args:
            dia_prod (float): dia_prod
            dia_incell (float): dia_incell
            thk_slit (float): thk_slit
            thk_outcell (float): thk_outcell
            thk_c2s (float): thk_c2s
            pitch_x (float): pitch_x
            pitch_y (float): pitch_y
            ratio_slit (int): ratio_slit

        Returns:
            incell (np.ndarray): ([X, Y])
            outcell (np.ndarray): ([X, Y])
        """
        # set outcell
        num = np.int64(np.ceil(dia_prod / pitch_x))
        _ = [(pitch_x * i, 0.0) for i in range(num)]
        [_.append((pitch_x * i, 0.0)) for i in range(-1, -num, -1)]
        outcell = np.array(_)

        # incell 1列目
        ref = np.copy(outcell)
        ref[:, 0] += 0.5 * pitch_x
        ref[:, 1] += 0.5 * max(thk_slit, thk_outcell) + thk_c2s + 0.5 * dia_incell

        # incell 2列目以降
        tmp = []
        tmp.append(ref)
        for i in range(1, ratio_slit):
            _ = np.copy(ref)
            if i % 2 == 0:
                _[:, 1] += pitch_y * i
            else:
                _[:, 0] += 0.5 * pitch_x
                _[:, 1] += pitch_y * i
            tmp.append(_)

        _ = np.array(tmp)
        incell = _.reshape((_.shape[0] * _.shape[1], _.shape[2]))

        return incell, outcell

    def copy_base_cell(
        pitch_x: float,
        ratio_slit: int,
        ce: np.ndarray,
        sl: np.ndarray,
    ) -> np.ndarray:
        """
        cell座標をY方向に複製

        Args:
            pitch_x (float): pitch_x
            ratio_slit (int): ratio_slit
            ce (np.ndarray): ([X, Y])
            sl (np.ndarray): ([Y])

        Returns:
            res (np.ndarray): ([X, Y])
        """
        sw = True
        tmp = []
        for y in sl:
            _ = np.copy(ce)
            if not sw:
                _[:, 0] += 0.5 * pitch_x
            _[:, 1] += y
            tmp.append(_)
            if ratio_slit % 2 == 0:
                sw = not sw

        _ = np.array(tmp)
        res = _.reshape((_.shape[0] * _.shape[1], _.shape[2]))

        return res

    def offset_xy(
        mode_cell: bool,
        mode_slit: bool,
        pitch_x: float,
        pitch_slit: float,
        ic: np.ndarray,
        oc: np.ndarray,
        sl: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        cellとslitのXY方向オフセット

        Args:
            mode_cell (bool): X方向オフセット(mode_cell)
            mode_slit (bool): Y方向オフセット(mode_slit)
            pitch_x (float): X方向オフセット量(pitch_x)
            pitch_slit (float): Y方向オフセット量(pitch_slit)
            ic (np.ndarray): incell ([X, Y])
            oc (np.ndarray): outcell ([X, Y])
            sl (np.ndarray): slit ([Y])

        Returns:
            incell (np.ndarray): ([X, Y])
            outcell (np.ndarray): ([X, Y])
            slit (np.ndarray): ([Y])
        """
        incell = np.copy(ic)
        outcell = np.copy(oc)
        slit = np.copy(sl)

        if mode_cell:
            x1 = 0.5 * pitch_x
            incell[:, 0] += x1
            outcell[:, 0] += x1

        if mode_slit:
            y1 = 0.5 * pitch_slit
            incell[:, 1] += y1
            outcell[:, 1] += y1
            slit = sl + y1

        return incell, outcell, slit

    def select_slit(
        lim_slit: float,
        sl: np.ndarray,
    ) -> np.ndarray:
        """
        lim_slit範囲内のslitを抽出

        Args:
            lim_slit (float): lim_slit
            sl (np.ndarray): slit ([Y])

        Returns:
            slit (np.ndarray): ([Y])
        """

        condition = abs(sl) <= lim_slit
        slit = sl[condition]

        return slit

    def select_cell(
        lim: float,
        ce: np.ndarray,
    ) -> np.ndarray:
        """
        限界値(lim_incell, lim_outcell)範囲内のcellを抽出

        Args:
            lim (float): 限界値(lim_incell, lim_outcell)
            ce (np.ndarray): ([X, Y])

        Returns:
            cell (np.ndarray): ([X, Y])
        """
        condition = np.sqrt(ce[:, 0] ** 2.0 + ce[:, 1] ** 2.0) <= lim
        cell = ce[condition]

        return cell


class Post:
    def draw(
        inp: tuple,
        ic: np.ndarray,
        oc: np.ndarray,
        sl: np.ndarray,
        fig_x: int = 1920,
        fig_y: int = 1080,
        fig_dpi: int = 100,
    ):
        # set fig, gridspec
        fig = plt.figure(figsize=(fig_x / fig_dpi, fig_y / fig_dpi), dpi=fig_dpi)
        gs = GridSpec(1, 6)
        ss1 = gs.new_subplotspec((0, 0), rowspan=1, colspan=3)
        ss2 = gs.new_subplotspec((0, 3), rowspan=1, colspan=3)

        # set axes
        ax1 = plt.subplot(ss1)
        ax2 = plt.subplot(ss2)

        ax1.grid(linewidth=0.2)
        ax1.set_axisbelow(True)
        ax1.set_aspect("equal", adjustable="datalim")

        scale = 0.52 * inp.dia_prod
        ax1.set_xlim(-scale, scale)
        ax1.set_ylim(-scale, scale)

        # product
        Post.product(ax1, inp.dia_prod, facecolor="None")
        Post.product(ax1, inp.dia_prod, transparency=0.5)
        Post.product(ax1, inp.dia_prod - 2.0 * inp.thk_prod, facecolor="None", transparency=1.0, linestyle="dashed")

        # slit
        Post.slit(ax1, inp.dia_prod, inp.thk_slit, sl)

        # cell
        Post.incell(ax1, inp.dia_incell - inp.thk_top - inp.thk_mid - inp.thk_bot, ic)
        Post.outcell(ax1, inp.thk_x1, inp.thk_x2, inp.thk_y1, inp.thk_y2, oc)

        # table
        _, _ = Post.table_calc(inp, ic, oc, sl)
        # Post.table(ax2, inp._asdict())

        plt.show()
        plt.close(fig)

    def product(
        ax: Axes,
        diameter: float,
        facecolor: str = "#BFBFBF",
        transparency: float = 1.0,
        linestyle: str = "solid",
    ) -> None:
        """
        製品径の描画(patches.Circle)

        Args:
            ax (Axes): Axes
            diameter (float): diameter
            facecolor (str, optional): facecolor. Defaults to "#BFBFBF".
            transparency (float, optional): linewidth. Defaults to 0.5.
            linestyle (str, optional): linestyle. Defaults to "solid".
        """
        _ = patches.Circle(
            xy=(0.0, 0.0),
            radius=0.5 * diameter,
            facecolor=facecolor,
            alpha=transparency,
            edgecolor="black",
            linestyle=linestyle,
        )
        ax.add_patch(_)

    def slit(
        ax: Axes,
        dia_prod: float,
        thk_slit: float,
        sl: np.ndarray,
    ) -> None:
        """
        slitの描画(plt.Polygon)

        Args:
            ax (Axes): Axes
            dia_prod (float): dia_prod
            thk_slit (float): thk_slit
            sl (np.ndarray): ([Y])
        """
        for y in sl:
            # slit淵右下(p1),右上(p2)の座標算出
            rrr = 0.5 * dia_prod
            y_t = y + 0.5 * thk_slit
            y_b = y - 0.5 * thk_slit
            x_t = np.sqrt(rrr**2.0 - y_t**2.0)
            x_b = np.sqrt(rrr**2.0 - y_b**2.0)
            p1 = np.array([x_b, y_b])
            p2 = np.array([x_t, y_t])

            # p1, p2のなす角を100分割した配列
            angle = np.arctan2(np.linalg.det([p1, p2]), np.dot(p1, p2))
            angles = np.linspace(0, angle, 100) + np.arctan2(p1[1], p1[0])

            # 円上の各角度におけるXY座標の算出(+X側->-X側)
            x_plus = []
            [x_plus.append([rrr * np.cos(ag), rrr * np.sin(ag)]) for ag in angles]
            x_plus = np.array(x_plus)
            x_minus = np.copy(x_plus)
            x_minus[:, 0] *= -1
            point = np.vstack([x_plus, x_minus])

            # 各点を角度順にソート
            angles = np.arctan2(point[:, 1], point[:, 0])
            sorted_point = point[np.argsort(angles)]

            # 描画
            _ = plt.Polygon(sorted_point, closed=True, facecolor="#97B6D8", fill=True, linewidth=0, alpha=0.5)
            ax.add_patch(_)

    def incell(
        ax: Axes,
        diameter: float,
        ic: np.ndarray,
    ) -> None:
        """
        incellの描画(patches.Circle)

        Args:
            ax (Axes): Axes
            diameter (float): diameter
            ic (np.ndarray): incell ([X, Y])
        """
        [ax.add_patch(patches.Circle(xy=_, radius=0.5 * diameter, facecolor="#4F81BD")) for _ in ic]

    def outcell(
        ax: Axes,
        thk_x1: float,
        thk_x2: float,
        thk_y1: float,
        thk_y2: float,
        oc: np.ndarray,
    ) -> None:
        """
        outcellの描画(plt.Polygon)

        Args:
            ax (Axes): Axes
            thk_x1 (float): thk_x1
            thk_x2 (float): thk_x2
            thk_y1 (float): thk_y1
            thk_y2 (float): thk_y2
            oc (np.ndarray): outcell ([X, Y])
        """
        # 八角形右上2点座標算出
        ref = np.array([(thk_x1 + thk_x2, thk_y2), (thk_x2, thk_y1 + thk_y2)])

        # X方向対称コピー
        tmp1 = np.copy(ref)
        tmp1[:, 0] *= -1.0
        tmp1 = np.vstack([ref, tmp1])

        # Y方向対称コピー
        tmp2 = np.copy(tmp1)
        tmp2[:, 1] *= -1.0
        point = np.vstack([tmp1, tmp2])

        # 各点を角度順にソート
        angles = np.arctan2(point[:, 1], point[:, 0])
        octagon_point = point[np.argsort(angles)]

        # 八角形の中心座標でオフセットして描画
        for x, y in oc:
            ref = np.copy(octagon_point)
            ref[:, 0] += x
            ref[:, 1] += y
            _ = plt.Polygon(ref, closed=True, facecolor="#76913C", fill=True, linewidth=0)
            ax.add_patch(_)

    def table_calc(
        inp: tuple,
        ic: np.ndarray,
        oc: np.ndarray,
        sl: np.ndarray,
    ) -> tuple[dict, dict]:
        # filtered_dict
        target = (
            "dia_incell",
            "thk_bot",
            "thk_mid",
            "thk_top",
            "thk_wall",
            "thk_c2s",
            "dia_prod",
            "thk_prod",
            "thk_outcell",
            "thk_wall_outcell",
            "thk_slit",
            "ratio_slit",
            "mode_cell",
            "mode_slit",
        )
        filtered_dict = {key: inp._asdict()[key] for key in target}

        ln_prod = 1000.0

        diameter_effective = inp.dia_incell - 2.0 * (inp.thk_top + inp.thk_mid + inp.thk_bot)
        area_incell = 0.25 * (np.pi * diameter_effective**2.0)
        area_outcell = 4.0 * ((inp.thk_x1 + inp.thk_x2) * (inp.thk_y1 + inp.thk_y2) - 0.5 * (inp.thk_x1 * inp.thk_y1))
        area_prod = 0.25 * (np.pi * inp.dia_prod**2.0)
        volume_incell = area_incell * ln_prod * ic.shape[0]
        volume_outcell = area_outcell * ln_prod * oc.shape[0]
        volume_prod = area_prod * ln_prod - volume_incell - volume_outcell

        calc = {
            "N(incell)": ic.shape[0],
            "N(outcell)": oc.shape[0],
            "N(slit)": sl.shape[0],
            "A(membrane)": np.pi * diameter_effective * ln_prod * ic.shape[0],
            "A(incell)": area_incell * ic.shape[0],
            "A(outcell)": area_outcell * oc.shape[0],
            "R_A(incell/prod)": area_incell / area_prod,
            "R_A(outcell/prod)": area_outcell / area_prod,
            "V(incell)": volume_incell,
            "V(outcell)": volume_outcell,
            "R_V(incell/prod)": volume_incell / volume_prod,
            "R_V(outcell/prod)": volume_outcell / volume_prod,
        }
        pp(calc, sort_dicts=False)

        return filtered_dict, calc

    def table(
        ax: Axes,
        inp_dict: dict,
    ) -> None:
        # val = []
        # key = []
        # # table_a
        # for k, v in self.res.inp.__dict__.items():
        #     if type(v) is np.float64:
        #         _ = [f"{v:.3e} [mm]"]
        #     elif type(v) is np.int64:
        #         _ = [f"{v} [-]"]
        #     else:
        #         _ = [f"{v}"]
        #     val.append(_)
        #     key.append(k)

        # # table_b
        # for k, v in self.table_b.items():
        #     if re.match("^N", k):
        #         _ = [f"{v} [-]"]
        #     elif re.match("^A", k):
        #         _ = [f"{v:.3e} [mm2]"]
        #     elif re.match("^R_A", k):
        #         _ = [f"{v*100:.1f} [%]"]
        #     elif re.match("^V", k):
        #         _ = [f"{v:.3e} [mm3]"]
        #     elif re.match("^R_V", k):
        #         _ = [f"{v*100:.1f} [%]"]
        #     val.append(_)
        #     key.append(k)

        # return val, key

        # # table
        # val, key = table()

        # # ax2 table
        # ax2 = plt.subplot(ss2)
        # tab = ax2.table(cellText=val, rowLabels=key, loc="center", colWidths=[1, 1])
        # for _, cell in tab.get_celld().items():
        #     cell.set_height(1 / len(val))
        # ax2.axis("off")
        # tab.set_fontsize(16)

        key = []
        val = []
        for k, v in inp_dict.items():
            key.append(str(k))
            val.append([str(v)])

        # ax.table(cellText=val, rowLabels=key, loc="center", colWidths=[1, 1])
        ax.table(cellText=val, rowLabels=key, loc="center")
        ax.axis("off")


if __name__ == "__main__":
    main()

# %%
