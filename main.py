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


def main():
    # コマンドライン引数を確認
    # if len(sys.argv) != 2:
    #     return None

    # 保存フォルダ作成
    # path = make_dir()
    # if path is None:
    #     return None

    # CSV読込み
    # params = read_csv(sys.argv[1])
    params = read_csv("inp.csv")

    # main
    for row in params.itertuples():
        # set slit
        base_slit = CirOctCalc.set_base_slit(dia_prod=row.dia_prod, pitch_slit=row.pitch_slit)

        # set base_cell
        base_incell, base_outcell = CirOctCalc.set_base_cell(
            dia_prod=row.dia_prod,
            dia_incell=row.dia_incell,
            thk_outcell=row.thk_outcell,
            thk_c2s=row.thk_c2s,
            pitch_x=row.pitch_x,
            pitch_y=row.pitch_y,
            ratio_slit=row.ratio_slit,
        )

        # copy base_cell
        copied_incell = CirOctCalc.copy_ycoord_xy(
            pitch_x=row.pitch_x,
            ratio_slit=row.ratio_slit,
            xy=base_incell,
            sl=base_slit[:, 1],
        )
        copied_outcell = CirOctCalc.copy_ycoord_xy(
            pitch_x=row.pitch_x,
            ratio_slit=row.ratio_slit,
            xy=base_outcell,
            sl=base_slit[:, 1],
        )

        # offset cell, slit
        offset_slit = CirOctCalc.offset_xy(
            mode_cell=False,
            mode_slit=row.mode_slit,
            pitch_x=0.0,
            pitch_slit=row.pitch_slit,
            xy=base_slit,
        )
        offset_incell = CirOctCalc.offset_xy(
            mode_cell=row.mode_cell,
            mode_slit=row.mode_slit,
            pitch_x=row.pitch_x,
            pitch_slit=row.pitch_slit,
            xy=copied_incell,
        )
        offset_outcell = CirOctCalc.offset_xy(
            mode_cell=row.mode_cell,
            mode_slit=row.mode_slit,
            pitch_x=row.pitch_x,
            pitch_slit=row.pitch_slit,
            xy=copied_outcell,
        )

        # select cell, slit
        select_slit = CirOctCalc.select_y(lim=row.lim_slit, xy=offset_slit)
        select_inell = CirOctCalc.select_r(lim=row.lim_incell, xy=offset_incell)
        tmp_outell = CirOctCalc.select_r(lim=row.lim_outcell, xy=offset_outcell)
        select_outell = CirOctCalc.select_y(lim=row.lim_slit, xy=tmp_outell)

        # draw
        CirOctPost.draw(inp=row, ic=select_inell, oc=select_outell, sl=select_slit)


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


class CirOctCalc:
    incell_shape: str = "cicle"
    outcell_shape: str = "octagon"

    @staticmethod
    def set_base_slit(
        dia_prod: float,
        pitch_slit: float,
    ) -> np.ndarray:
        """
        slitの基準座標(X, Y)を算出

        Args:
            dia_prod (float): dia_prod
            pitch_slit (float): pitch_slit

        Returns:
            res (np.ndarray): (X, Y)
        """
        num = np.int64(np.ceil(dia_prod / pitch_slit))

        # +Y方向配置(0<=)
        tmp_slit = [(0.0, pitch_slit * i) for i in range(num)]

        # -Y方向配置(<0)
        [tmp_slit.append((0.0, pitch_slit * i)) for i in range(-1, -num, -1)]

        # sort
        tmp_slit = np.array(tmp_slit)
        res = tmp_slit[tmp_slit[:, 1].argsort()]

        return res

    @staticmethod
    def set_base_cell(
        dia_prod: float,
        dia_incell: float,
        thk_outcell: float,
        thk_c2s: float,
        pitch_x: float,
        pitch_y: float,
        ratio_slit: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        incellとoutcellの基準座標(X, Y)を算出

        Args:
            dia_prod (float): dia_prod
            dia_incell (float): dia_incell
            thk_outcell (float): thk_outcell
            thk_c2s (float): thk_c2s
            pitch_x (float): pitch_x
            pitch_y (float): pitch_y
            ratio_slit (int): ratio_slit

        Returns:
            incell (np.ndarray): (X, Y)
            outcell (np.ndarray): (X, Y)
        """
        # set outcell (Y=0.0)
        num = np.int64(np.ceil(dia_prod / pitch_x))
        oc = [(pitch_x * i, 0.0) for i in range(num)]
        [oc.append((pitch_x * i, 0.0)) for i in range(-1, -num, -1)]
        outcell = np.array(oc)

        # set incell 1列目
        row = np.copy(outcell)
        row[:, 0] += 0.5 * pitch_x
        row[:, 1] += 0.5 * (thk_outcell + dia_incell) + thk_c2s

        # set incell 2列目以降
        incell = np.copy(row)
        for i in range(1, ratio_slit):
            ic = np.copy(row)
            if i % 2 != 0:
                ic[:, 0] += 0.5 * pitch_x
            ic[:, 1] += pitch_y * i
            incell = np.vstack([incell, ic])

        return incell, outcell

    @staticmethod
    def copy_ycoord_xy(
        pitch_x: float,
        ratio_slit: int,
        xy: np.ndarray,
        sl: np.ndarray,
    ) -> np.ndarray:
        """
        cell座標をY方向に複製

        Args:
            pitch_x (float): pitch_x
            ratio_slit (int): ratio_slit
            xy (np.ndarray): (X, Y)
            sl (np.ndarray): (Y)

        Returns:
            res (np.ndarray): (X, Y)
        """
        size = len(sl)
        if ratio_slit % 2 == 0:
            switches = np.tile([True, False], size // 2 + 1)[:size]
            if not switches[np.where(sl == 0.0)[0]]:
                switches = ~switches
        else:
            switches = np.tile([True, True], size // 2 + 1)[:size]

        # +Y方向コピー(0<=)
        plus_xy = np.copy(xy)
        for y, sw in zip(sl, switches):
            copied_xy = np.copy(xy)
            if not sw:
                copied_xy[:, 0] += 0.5 * pitch_x
            copied_xy[:, 1] += y
            plus_xy = np.vstack([plus_xy, copied_xy])

        # -Y方向コピー(<0)
        condition = plus_xy[:, 1] > 0.0
        tmp_xy = np.copy(plus_xy[condition])
        tmp_xy[:, 1] *= -1.0
        res = np.vstack([plus_xy, tmp_xy])

        return res

    @staticmethod
    def offset_xy(
        mode_cell: bool,
        mode_slit: bool,
        pitch_x: float,
        pitch_slit: float,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        cellとslitのXY方向オフセット

        Args:
            mode_cell (bool): X方向スイッチ(mode_cell)
            mode_slit (bool): Y方向スイッチ(mode_slit)
            pitch_x (float): X方向オフセット量(pitch_x)
            pitch_slit (float): Y方向オフセット量(pitch_slit)
            xy (np.ndarray): 座標(cell or slit) (X, Y)

        Returns:
            res (np.ndarray): (X, Y)
        """
        res = np.copy(xy)

        if mode_cell:
            res[:, 0] += 0.5 * pitch_x

        if mode_slit:
            res[:, 1] += 0.5 * pitch_slit

        return res

    @staticmethod
    def select_r(
        lim: float,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        (X, Y)座標->半径Rが範囲内の要素を抽出

        Args:
            lim (float): 限界値(lim_incell, lim_outcell)
            xy (np.ndarray): ([X, Y])

        Returns:
            (np.ndarray): ([X, Y])
        """
        condition = np.sqrt(xy[:, 0] ** 2.0 + xy[:, 1] ** 2.0) <= lim

        return xy[condition]

    @staticmethod
    def select_y(
        lim: float,
        xy: np.ndarray,
    ) -> np.ndarray:
        """
        (X, Y)座標のYが範囲内の要素を抽出

        Args:
            lim (float): 限界値(lim_incell, lim_outcell)
            xy (np.ndarray): ([X, Y])

        Returns:
            (np.ndarray): ([X, Y])
        """
        condition = abs(xy[:, 1]) <= lim

        return xy[condition]


class CirOctPost:
    @classmethod
    def draw(
        cls,
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
        gs = GridSpec(1, 8)
        ss1 = gs.new_subplotspec((0, 0), colspan=5)
        ss2 = gs.new_subplotspec((0, 6), colspan=2)

        # set ax1
        ax1 = plt.subplot(ss1)

        ax1.grid(linewidth=0.2)
        ax1.set_axisbelow(True)
        ax1.set_aspect("equal", adjustable="datalim")

        scale = 0.52 * inp.dia_prod
        ax1.set_xlim(-scale, scale)
        ax1.set_ylim(-scale, scale)

        # set ax2
        ax2 = plt.subplot(ss2)
        ax2.axis("off")

        # product
        cls._product(ax1, inp.dia_prod, facecolor="None")
        cls._product(ax1, inp.dia_prod, transparency=0.5)
        cls._product(ax1, inp.dia_prod - 2.0 * inp.thk_prod, facecolor="None", transparency=1.0, linestyle="dashed")

        # slit
        cls._slit(ax1, inp.dia_prod, inp.thk_slit, sl[:, 1])

        # cell
        cls._incell(ax1, inp.dia_incell - inp.thk_top - inp.thk_mid - inp.thk_bot, ic)
        cls._outcell(ax1, inp.thk_x1, inp.thk_x2, inp.thk_y1, inp.thk_y2, oc)

        # table
        cls._table(ax2, cls._table_calc(inp._asdict(), ic, oc, sl))

        plt.show()
        plt.close(fig)

    def _product(
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

    def _slit(
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
            # vertex(円弧10分割)の座標算出
            radius = 0.5 * dia_prod
            y_t = y + 0.5 * thk_slit
            y_b = y - 0.5 * thk_slit
            theta1 = np.arcsin(y_b / radius)
            theta2 = np.arcsin(y_t / radius)
            angles = np.linspace(theta1, theta2, 10)
            xy = np.array([radius * np.cos(angles), radius * np.sin(angles)]).T

            # -X方向コピー
            copied_vertex = np.copy(xy)
            copied_vertex[:, 0] *= -1
            vertex = np.vstack([xy, copied_vertex])

            # vertexを角度順にソート
            angles = np.arctan2(vertex[:, 1], vertex[:, 0])
            sorted_vertex = vertex[np.argsort(angles)]

            # 描画
            _ = plt.Polygon(sorted_vertex, closed=True, facecolor="#97B6D8", fill=True, linewidth=0, alpha=0.5)
            ax.add_patch(_)

    def _incell(
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

    def _outcell(
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

    def _table_calc(
        inp: dict,
        ic: np.ndarray,
        oc: np.ndarray,
        sl: np.ndarray,
    ) -> dict:
        """
        計算結果画像に記載するパラメータ表の作成

        Args:
            inp (dict): input parameters
            ic (np.ndarray): incell ([X, Y])
            oc (np.ndarray): outcell ([X, Y])
            sl (np.ndarray): slit ([X, Y])

        Returns:
            res (dict): valueに単位を付けた文字列辞書
        """
        # 画像に出力するパラメータ
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
        filtered_dict = {key: inp[key] for key in target}

        # 値を文字列化
        val = []
        key = []

        for k, v in filtered_dict.items():
            if "dia_" in k or "thk_" in k:
                string = f"{v:.2f} [mm]"
            else:
                string = f"{v}"
            val.append(string)
            key.append(k)

        res = dict(zip(key, val))

        # セル数等の計算
        thk_mem = inp["thk_top"] + inp["thk_mid"] + inp["thk_bot"]
        diameter_effective = inp["dia_incell"] - 2.0 * thk_mem
        area_incell = 0.25 * (np.pi * diameter_effective**2.0)
        area_prod = 0.25 * (np.pi * inp["dia_prod"] ** 2.0)
        area_mem = np.pi * diameter_effective * inp["ln_prod"] * ic.shape[0]

        res["N(incell)"] = f"{ic.shape[0]}"
        res["N(outcell)"] = f"{oc.shape[0]}"
        res["N(slit)"] = f"{sl.shape[0]}"
        res["A(membrane)"] = f"{area_mem:.2e} [mm2]"
        res["A(incell)"] = f"{(area_incell * ic.shape[0]):.2e} [mm2]"
        res["R_A(incell/prod)"] = f"{(area_incell * ic.shape[0] / area_prod*100.0):.1f} [%]"

        return res

    def _table(
        ax: Axes,
        inp: dict,
    ) -> None:
        """
        パラメータ表の描画

        Args:
            ax (Axes): Axes
            inp (dict): input parameters
        """
        key = [str(k) for k in inp.keys()]
        val = [[str(v)] for v in inp.values()]

        tab = ax.table(cellText=val, rowLabels=key, loc="center")
        tab.set_fontsize(16)
        for _, cell in tab.get_celld().items():
            cell.set_height(1 / len(val))


if __name__ == "__main__":
    main()
