"""
title   : cell_layout_v4
ver     : 4.0
update  : 2023/05/11

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

import json
import re
from pathlib import Path
from datetime import datetime
from itertools import product, zip_longest
from dataclasses import dataclass
from typing import get_type_hints


def json_to_dict(path: str) -> dict:
    """
    jsonファイルの読み込み

    Parameters
    ----------
    path : str
        jsonfile path

    Returns
    -------
    res : dict
        jsonを正常に読み込んだ辞書

    Raises
    ------
    FileNotFoundError
        jsonfileの有無
    JSONDecodeError
        jsonfileのフォーマット異常
    """
    try:
        if Path(path).exists():
            with open(path, "r") as f:
                res = json.load(f)
            return res
        else:
            raise FileNotFoundError

    except FileNotFoundError:
        print("input.json not found.")
        return None

    except json.JSONDecodeError as e:
        print(f"json format eroor : {e}")
        return None


def dict_to_df(inp: dict) -> pd.DataFrame:
    """
    辞書型データの直積をDataFrame型に変換

    Parameters
    ----------
    inp : dict
        json_to_dictの戻り値を想定したデータ

    Returns
    -------
    res : pd.DataFrame
        DataFrame型で水準一覧を返す
    """
    _ = list(product(*inp.values()))
    res = pd.DataFrame(_, columns=[k for k in inp.keys()])
    return res


def make_dir() -> list:
    """
    結果保存用フォルダの作成

    Returns
    -------
    list : WindowsPath
        0 : カレントフォルダ
        1 : 座標データフォルダ

    Raises
    ------
    Exception
        同名フォルダが存在した場合
    """
    dir_path = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    dir_pos = dir_path / Path("positions_data")
    try:
        if not dir_path.exists():
            dir_path.mkdir()
            dir_pos.mkdir()
            return [dir_path, dir_pos]
        else:
            raise Exception
    except Exception:
        print("Exception")
        return None


def except_case(cls: Exception) -> pd.Series:
    tmp = {
        "dia_incell": "None",
        "thk_bot": "None",
        "thk_mid": "None",
        "thk_top": "None",
        "thk_outcell": "None",
        "thk_wall_outcell": "None",
        "dia_prod": "None",
        "thk_prod": "None",
        "thk_wall": "None",
        "thk_c2s": "None",
        "ln_prod": "None",
        "ln_slit": "None",
        "ln_edge": "None",
        "ln_glass_seal": "None",
        "thk_slit": "None",
        "ratio_slit": "None",
        "mode_cell": "None",
        "mode_slit": "None",
        "pitch_x": "None",
        "pitch_y": "None",
        "pitch_slit": "None",
        "thk_x1": "None",
        "thk_y1": "None",
        "lim_slit": "None",
        "lim_incell": "None",
        "lim_outcell": "None",
        "N(incell)": "None",
        "N(outcell)": "None",
        "N(slit)": "None",
        "A(membrane)": "None",
        "A(incell)": "None",
        "A(outcell)": "None",
        "R_A(incell/product)": "None",
        "R_A(outcell/product)": "None",
        "V(incell)": "None",
        "V(outcell)": "None",
        "R_V(incell/product)": "None",
        "R_V(outcell/product)": "None",
        "ERROR": cls.args[0],
    }
    res = pd.Series(tmp)
    return res


@dataclass
class CirOct:
    """
    incell : circle
    outcell : octagon

    Raises
    ------
    TypeError
        input.jsonに記載された値が, 想定していない型だった場合
    ValueError
        CirOctValidationクラスメソッドで定義
    """

    # incell
    dia_incell: np.float64 = None
    # emmbrane
    thk_bot: np.float64 = None
    thk_mid: np.float64 = None
    thk_top: np.float64 = None
    # outcell
    thk_outcell: np.float64 = None
    thk_wall_outcell: np.float64 = None
    # product
    dia_prod: np.float64 = None
    thk_prod: np.float64 = None
    thk_wall: np.float64 = None
    thk_c2s: np.float64 = None
    ln_prod: np.float64 = None
    ln_slit: np.float64 = None
    ln_edge: np.float64 = None
    ln_glass_seal: np.float64 = None
    thk_slit: np.float64 = None
    ratio_slit: np.int64 = None
    mode_cell: np.bool_ = None
    mode_slit: np.bool_ = None
    # calc
    pitch_x: np.float64 = None
    pitch_y: np.float64 = None
    pitch_slit: np.float64 = None
    thk_x1: np.float64 = None
    thk_y1: np.float64 = None
    lim_slit: np.float64 = None
    lim_incell: np.float64 = None
    lim_outcell: np.float64 = None

    def __post_init__(self):
        """
        インスタンス化された際に入力値の型を確認

        """
        # TypeError Check
        class_type_hints = get_type_hints(CirOct)
        class_vars = [_ for _ in dir(self) if not callable(getattr(self, _)) and not _.startswith("__")]
        for attr_name in class_vars:
            tmp1 = class_type_hints[attr_name]
            tmp2 = type(getattr(self, attr_name))
            if tmp2 is not tmp1 and tmp2 != type(None):
                raise TypeError(f"{attr_name} must be {tmp1.__name__} or NoneType, not {tmp2.__name__}")

        # calc
        CirOctCalc.pitch_x(self)
        CirOctCalc.pitch_y(self)
        CirOctCalc.pitch_slit(self)
        CirOctCalc.thk_x1(self)
        CirOctCalc.thk_y1(self)
        CirOctCalc.lim_slit(self)
        CirOctCalc.lim_incell(self)
        CirOctCalc.lim_outcell(self)

        # ValueError Check
        CirOctValidation.ve01(self)
        CirOctValidation.ve02(self)
        CirOctValidation.ve03(self)
        CirOctValidation.ve04(self)
        # CirOctValidation.ve05(self)
        # CirOctValidation.ve06(self)
        # CirOctValidation.ve07(self)


class CirOctCalc:
    """
    設計変数から算出するためのメソッド

    """

    def pitch_x(cls: CirOct):
        """
        インセルX方向ピッチを算出

        """
        cls.pitch_x = cls.dia_incell + cls.thk_wall

    def pitch_y(cls: CirOct):
        """
        インセルY方向ピッチを算出

        """
        cls.pitch_y = cls.pitch_x * np.sin(np.pi / 3.0)

    def pitch_slit(cls: CirOct):
        """
        スリットピッチを算出

        """
        tmp = max(cls.thk_slit, cls.thk_outcell) + cls.dia_incell + 2.0 * cls.thk_c2s
        cls.pitch_slit = tmp + (cls.ratio_slit - 1) * cls.pitch_y

    def thk_x1(cls: CirOct, scale: np.float64 = 0.345):
        """
        アウトセル角部のX方向寸法の算出

        Parameters
        ----------
        scale : np.float64
            アウトセル角部寸法の算出用係数, by default 0.345
        """
        tmp = 0.5 * (cls.pitch_x - cls.thk_wall_outcell)
        cls.thk_x1 = scale * tmp

    def thk_y1(cls: CirOct, scale: np.float64 = 0.300):
        """
        アウトセル角部のY方向寸法の算出

        Parameters
        ----------
        scale : np.float64
            アウトセル角部寸法の算出用係数, by default 0.300
        """
        tmp = 0.5 * cls.thk_outcell
        cls.thk_y1 = scale * tmp

    def lim_slit(cls: CirOct):
        """
        スリット位置のY方向限界値を算出
        Y方向末端に最低1列のインセルを置く

        """
        tmp = 0.5 * (cls.dia_prod - max(cls.thk_slit, cls.thk_outcell))
        cls.lim_slit = tmp - cls.thk_prod - cls.dia_incell - cls.thk_c2s

    def lim_incell(cls: CirOct):
        """
        製品中心からインセルを配置する制限半径を算出

        """
        cls.lim_incell = 0.5 * cls.dia_prod - cls.thk_prod - 0.5 * cls.dia_incell

    def lim_outcell(cls: CirOct):
        """
        製品中心からアウトセルを配置する制限半径を算出

        """
        dx = 0.5 * (cls.pitch_x - cls.thk_wall_outcell)
        dy = 0.5 * cls.thk_outcell
        aaa = np.sqrt(dx**2 + (dy - cls.thk_y1) ** 2)
        bbb = np.sqrt((dx - cls.thk_x1) ** 2 + dy**2)
        cls.lim_outcell = 0.5 * cls.dia_prod - cls.thk_prod - max(aaa, bbb)


class CirOctValidation:
    """
    設計変数を検証するためのメソッド

    """

    def ve01(cls: CirOct):
        """
        論理型以外のクラス変数に負の値があるか判断
        thk_wall_outcell, thk_c2sが負の場合, dia_incellと同値

        Raises
        ------
        ValueError
            クラス変数に負の値がある場合
        """
        class_vars = [_ for _ in dir(cls) if not callable(getattr(cls, _)) and not _.startswith("__")]
        for attr_name in class_vars:
            tmp = getattr(cls, attr_name)
            if type(tmp) is not np.bool_:
                if tmp < 0:
                    if "thk_wall_outcell" in attr_name:
                        cls.thk_wall_outcell = cls.thk_wall
                    elif "thk_c2s" in attr_name:
                        cls.thk_c2s = cls.thk_wall
                    else:
                        raise ValueError(f"ValueError : VE01 {attr_name} is negative.")

    def ve02(cls: CirOct):
        """
        dia_incell <= 2*(thk_bot + thk_mid + thk_top) -> True
        インセル有効径が0でも動作する

        Raises
        ------
        ValueError
            層厚の合計が大きすぎる場合
        """
        if cls.dia_incell <= 2.0 * (cls.thk_bot + cls.thk_mid + cls.thk_top):
            raise ValueError("ValueError : VE02")

    def ve03(cls: CirOct):
        """
        ln_slitとln_glass_sealの関係を制限(全スリットは注意が必要)

        Raises
        ------
        ValueError
            VE06-1
            ln_prod <= 2*(ln_glass_seal + ln_slit) -> True
        ValueError
            VE06-2
            ln_glass_seal < ln_edge -> True
        """
        if cls.ln_prod <= 2.0 * (cls.ln_glass_seal + cls.ln_slit):
            raise ValueError("ValueError : VE03-1")
        if cls.ln_glass_seal > cls.ln_edge:
            raise ValueError("ValueError : VE03-2")

    def ve04(cls: CirOct):
        """
        dia_prod, thk_prodの制限

        Raises
        ------
        ValueError
            VE07-1
            dia_prod <= 2*thk_prod -> True
        ValueError
            VE07-2
            (dia_prod - 2*thk_prod) < dia_incell -> True
        """
        if cls.dia_prod <= 2.0 * cls.thk_prod:
            raise ValueError("ValueError : VE04-1")
        if (cls.dia_prod - 2.0 * cls.thk_prod) < cls.dia_incell:
            raise ValueError("ValueError : VE04-2")

    def ve05(cls: CirOct, lim: np.float64 = 0.1):
        """
        アウトセルの微小エッジを制限
        斜辺長さ
        下限値は未検討(202305)

        Parameters
        ----------
        lim : np.float64
            下限値, by default 0.1

        Raises
        ------
        ValueError
            面取り部が下限値以下の場合
        """
        if np.sqrt(cls.thk_x1**2 + cls.thk_y1**2) <= lim:
            raise ValueError("ValueError : VE05")

    def ve06(cls: CirOct, lim: np.float64 = 0.1):
        """
        アウトセルの微小エッジを制限
        XY方向寸法
        下限値は未検討(202305)

        Parameters
        ----------
        lim : np.float64
            下限値, by default 0.1

        Raises
        ------
        ValueError
            VE04-1
            アウトセルX方向寸法が下限値以下の場合
        ValueError
            VE04-2
            アウトセルY方向寸法が下限値以下の場合
        """
        wid_outcell = cls.pitch_x - cls.thk_wall_outcell
        tmp1 = wid_outcell - 2.0 * cls.thk_x1
        tmp2 = cls.thk_outcell - 2.0 * cls.thk_y1
        if tmp1 < lim:
            raise ValueError(f"ValueError : VE06-1 {tmp1}")
        if tmp2 < lim:
            raise ValueError(f"ValueError : VE06-2 {tmp2}")

    def ve07(cls: CirOct, lim: np.float64 = 0.1):
        """
        アウトセルの微小エッジを制限
        下限値は未検討(202305)

        Parameters
        ----------
        lim : np.float64
            下限値, by default 0.1

        Raises
        ------
        ValueError
            VE05-1
            thk_outcellとthk_slitの差が下限値以下の場合
        ValueError
            VE05-2
            アウトセルY方向寸法が下限値以下の場合
        ValueError
            VE05-3
            thk_outcellとthk_slitの差が下限値以下の場合
        """
        tmp1 = abs(cls.thk_outcell - cls.thk_slit)
        if cls.thk_outcell >= cls.thk_slit:
            tmp2 = 0.5 * (cls.thk_slit - (cls.thk_outcell - 2.0 * cls.thk_y1))
            if 0.0 < tmp1 <= lim:
                raise ValueError(f"ValueError : VE07-2 {tmp1}")
            if 0.0 < tmp2 <= lim:
                raise ValueError(f"ValueError : VE07-1 {tmp2}")
        else:
            if 0.0 < tmp1 < lim:
                raise ValueError(f"ValueError : VE07-3 {tmp1}")


@dataclass
class Result:
    """
    incell : circle
    outcell : octagon

    """

    # input parameters
    inp: CirOct
    # calc
    base_slit = np.empty([0, 2])
    base_cell = np.empty([0, 2])
    copy_incell = np.empty([0, 2])
    copy_outcell = np.empty([0, 2])
    pos_slit = np.empty([0, 2])
    pos_incell = np.empty([0, 2])
    pos_outcell = np.empty([0, 2])

    def __post_init__(self):
        """
        インセル, アウトセル, スリット座標を算出

        """
        ResultCalc.set_base_slit(self)
        ResultCalc.set_base_cell(self)
        ResultCalc.copy_base_cell(self)
        ResultCalc.offset_xy(self)
        ResultCalc.select_slit(self)
        ResultCalc.select_incell(self)
        ResultCalc.select_outcell(self)


class ResultCalc:
    """
    設計変数からインセル, アウトセル, スリット座標を算出するメソッド

    """

    def set_base_slit(cls: Result):
        """
        基準となるスリットのY座標を算出
        base_slit = array([Y, Z-length])

        """
        num = 5 * np.int64(cls.inp.dia_prod / cls.inp.pitch_slit)
        y1 = []
        # plus = [cls.inp.pitch_slit * i for i in range(num) if cls.inp.dia_prod >= cls.inp.pitch_slit * i]
        plus = [cls.inp.pitch_slit * i for i in range(num)]
        y1[len(y1) : len(y1)] = plus
        y1[len(y1) : len(y1)] = [-_ for _ in plus if _ > 0.0]

        y1.sort()
        _ = [[y, z] for y, z in zip_longest(y1, [cls.inp.ln_slit], fillvalue=cls.inp.ln_slit)]
        cls.base_slit = np.array(_)

    def set_base_cell(cls: Result):
        """
        インセルとアウトセルの1ピッチ分を算出
        base_cell = array([X, Y])

        """
        num = 2 * np.int64(cls.inp.dia_prod / cls.inp.pitch_x)
        x1 = []
        plus = [cls.inp.pitch_x * i for i in range(num)]
        x1[len(x1) : len(x1)] = plus
        x1[len(x1) : len(x1)] = [-x for x in plus if x > 0.0]

        x2 = [x + (0.5 * cls.inp.pitch_x) for x in x1]

        o2i_y = cls.inp.thk_c2s + 0.5 * (max(cls.inp.thk_slit, cls.inp.thk_outcell) + cls.inp.dia_incell)
        y1 = [0.0]
        y1[len(y1) : len(y1)] = [o2i_y + cls.inp.pitch_y * i for i in range(cls.inp.ratio_slit)]

        _ = []
        for i in range(len(y1)):
            if i % 2 == 0:
                [_.append([x, y1[i]]) for x in x1]
            else:
                [_.append([x, y1[i]]) for x in x2]
        cls.base_cell = np.array(_)

    def copy_base_cell(cls: Result):
        """
        Y方向にコピーして, インセルとアウトセルに配列を分ける
        copy_incell = array([X, Y])
        copy_outcell = array([X, Y])

        """
        yyy = cls.base_slit[:, 0].tolist()
        if (cls.inp.ratio_slit + 1) % 2 == 0:
            xxx = [0.0] * len(yyy)
        else:
            xxx = [0.5 * cls.inp.pitch_x if i % 2 != 0 else 0.0 for i in range(len(yyy))]

        ic = []
        oc = []
        for x1, y1 in zip(xxx, yyy):
            _ = [[x + x1, y + y1] for x, y in cls.base_cell.tolist() if y == 0.0]
            oc[len(oc) : len(oc)] = _
            _ = [[x + x1, y + y1] for x, y in cls.base_cell.tolist() if y != 0.0]
            ic[len(ic) : len(ic)] = _
        cls.copy_incell = np.array(ic)
        cls.copy_outcell = np.array(oc)

    def offset_xy(cls: Result):
        """
        (mode_cell, mode_slit)
        -> (True, True) : スリット中心 & アウトセル中心
        Falseで半ピッチずれる
        copy_incell = array([X, Y])
        copy_outcell = array([X, Y])
        base_slit = array([Y, Z])

        """
        if not cls.inp.mode_cell:
            x1 = 0.5 * cls.inp.pitch_x
            ic = [[x + x1, y] for x, y in cls.copy_incell.tolist()]
            oc = [[x + x1, y] for x, y in cls.copy_outcell.tolist()]
            cls.copy_incell = np.array(ic)
            cls.copy_outcell = np.array(oc)
        else:
            pass

        if not cls.inp.mode_slit:
            y1 = 0.5 * cls.inp.pitch_slit
            ic = [[x, y + y1] for x, y in cls.copy_incell.tolist()]
            oc = [[x, y + y1] for x, y in cls.copy_outcell.tolist()]
            sl = [[y + y1, z] for y, z in cls.base_slit.tolist()]
            cls.copy_incell = np.array(ic)
            cls.copy_outcell = np.array(oc)
            cls.base_slit = np.array(sl)
        else:
            pass

    def select_slit(cls: Result):
        """
        lim_slit範囲内のスリットを抽出

        pos_slit = array([Y, Z])

        """
        _ = [[y, z] for y, z in cls.base_slit.tolist() if cls.inp.lim_slit >= abs(y)]
        cls.pos_slit = np.array(_)

    def select_incell(cls: Result):
        """
        lim_incell範囲内のインセル(押出径)を抽出
        pos_incell = array([X, Y])

        """
        target = cls.copy_incell.tolist()
        _ = [[x, y] for x, y in target if cls.inp.lim_incell >= np.sqrt(x**2 + y**2)]
        cls.pos_incell = np.array(_)

    def select_outcell(cls: Result):
        """
        lim_outcell範囲内のアウトセルを抽出
        pos_outcell = array([X, Y])

        """
        target = cls.copy_outcell.tolist()
        _ = [[x, y] for x, y in target if cls.inp.lim_outcell >= np.sqrt(x**2 + y**2)]
        cls.pos_outcell = np.array(_)


@dataclass
class Post:
    """
    結果描画, 出力メソッド

    """

    res: Result
    path: list
    id: int
    table_b: dict = None

    def __post_init__(self):
        """
        評価変数の算出

        """
        self.table_b = {}
        eff_dia = self.res.inp.dia_incell - 2.0 * (self.res.inp.thk_bot + self.res.inp.thk_mid + self.res.inp.thk_top)
        eff_cir = np.pi * eff_dia
        area_incell = 0.25 * (np.pi * eff_dia**2)
        area_sq = self.res.inp.thk_outcell * (self.res.inp.pitch_x - self.res.inp.thk_wall_outcell)
        area_outcell = area_sq - 2.0 * (self.res.inp.thk_x1 * self.res.inp.thk_y1)
        area_prod = 0.25 * (np.pi * self.res.inp.dia_prod**2)
        ln_ocell = self.res.inp.ln_prod - 2.0 * (self.res.inp.ln_edge + self.res.inp.ln_slit)
        vol_prod = area_prod * self.res.inp.ln_prod

        self.table_b["N(incell)"] = self.res.pos_incell.shape[0]
        self.table_b["N(outcell)"] = self.res.pos_outcell.shape[0]
        self.table_b["N(slit)"] = self.res.pos_slit.shape[0]
        self.table_b["A(membrane)"] = eff_cir * self.res.inp.ln_prod * self.res.pos_incell.shape[0]
        self.table_b["A(incell)"] = area_incell * self.res.pos_incell.shape[0]
        self.table_b["A(outcell)"] = area_outcell * self.res.pos_outcell.shape[0]
        self.table_b["R_A(incell/product)"] = self.table_b["A(incell)"] / area_prod
        self.table_b["R_A(outcell/product)"] = self.table_b["A(outcell)"] / area_prod
        self.table_b["V(incell)"] = self.table_b["A(incell)"] * self.res.inp.ln_prod
        self.table_b["V(outcell)"] = self.table_b["A(incell)"] * ln_ocell
        self.table_b["R_V(incell/product)"] = self.table_b["V(incell)"] / vol_prod
        self.table_b["R_V(outcell/product)"] = self.table_b["V(outcell)"] / vol_prod

    def draw(self, dbg: bool = False, fig_x: int = 1920, fig_y: int = 1080, fig_dpi: int = 100):
        """
        画像ファイル(PNG)の出力

        Parameters
        ----------
        dbg : bool
            デバッグモードフラグ, by default False
        fig_x : int
            画像サイズ高さ, by default 1920
        fig_y : int
            画像サイズ横幅, by default 1080
        fig_dpi : int
            画像dpi, by default 100
        """

        def product(diameter: np.float64, style: str = "solid", width: float = 1):
            radius = 0.5 * diameter
            theta = np.linspace(0.0, 2.0 * np.pi, 360)
            xxx = 0.0 + radius * np.cos(theta)
            yyy = 0.0 + radius * np.sin(theta)

            ax1.plot(xxx, yyy, color="black", linestyle=style, linewidth=width)

        def slit(width: float = 0.6):
            if not dbg:
                yyy = [y for y, _ in self.res.pos_slit.tolist()]
            else:
                yyy = [y for y, _ in self.res.base_slit.tolist()]

            y_t = [yyy + 0.5 * self.res.inp.thk_slit] * 2
            y_b = [yyy - 0.5 * self.res.inp.thk_slit] * 2

            if not dbg:
                _ = (0.5 * self.res.inp.dia_prod) ** 2.0
                x_t = [-np.sqrt(_ - y_t[0] ** 2.0), np.sqrt(_ - y_t[1] ** 2.0)]
                x_b = [-np.sqrt(_ - y_b[0] ** 2.0), np.sqrt(_ - y_b[1] ** 2.0)]
            else:
                x_t = [-self.res.inp.dia_prod, self.res.inp.dia_prod]
                x_b = [-self.res.inp.dia_prod, self.res.inp.dia_prod]

            ax1.plot(x_t, y_t, color="blue", linewidth=width, linestyle="dashed")
            ax1.plot(x_b, y_b, color="blue", linewidth=width, linestyle="dashed")

        def incell():
            if not dbg:
                mem = self.res.inp.thk_bot + self.res.inp.thk_mid + self.res.inp.thk_top
                r_cell = 0.5 * self.res.inp.dia_incell - mem
                color = "tab:blue"
                target = self.res.pos_incell.tolist()
            else:
                r_cell = 0.5 * self.res.inp.dia_incell
                color = "gray"
                target = self.res.copy_incell.tolist()

            [ax1.add_patch(patches.Circle(xy=tuple(_), radius=r_cell, fc=color)) for _ in target]

        def outcell():
            ref = []
            dx = 0.5 * (self.res.inp.pitch_x - self.res.inp.thk_wall_outcell)
            dy = 0.5 * self.res.inp.thk_outcell
            ref.append([dx, dy - self.res.inp.thk_y1])
            ref.append([dx - self.res.inp.thk_x1, dy])
            ref.append([-ref[1][0], ref[1][1]])
            ref.append([-ref[0][0], ref[0][1]])
            ref.append([ref[3][0], -ref[3][1]])
            ref.append([ref[2][0], -ref[2][1]])
            ref.append([ref[1][0], -ref[1][1]])
            ref.append([ref[0][0], -ref[0][1]])

            _ = np.array(ref)
            ref = np.insert(_, [2], 1.0, axis=1)
            if not dbg:
                target = self.res.pos_outcell.tolist()
            else:
                target = self.res.copy_outcell.tolist()
            for xxx, yyy in target:
                tra = np.array([[1, 0, xxx], [0, 1, yyy], [0, 0, 1]])
                _ = np.array([tra @ xy for xy in ref])
                res = np.delete(_, [2], axis=1)
                _ = plt.Polygon(res, fc="green")
                ax1.add_patch(_)

        def table() -> list:
            val = []
            key = []
            # table_a
            for k, v in self.res.inp.__dict__.items():
                if type(v) is np.float64:
                    _ = [f"{v:.3e} [mm]"]
                elif type(v) is np.int64:
                    _ = [f"{v} [-]"]
                else:
                    _ = [f"{v}"]
                val.append(_)
                key.append(k)

            # table_b
            for k, v in self.table_b.items():
                if re.match("^N", k):
                    _ = [f"{v} [-]"]
                elif re.match("^A", k):
                    _ = [f"{v:.3e} [mm2]"]
                elif re.match("^R_A", k):
                    _ = [f"{v*100:.1f} [%]"]
                elif re.match("^V", k):
                    _ = [f"{v:.3e} [mm3]"]
                elif re.match("^R_V", k):
                    _ = [f"{v*100:.1f} [%]"]
                val.append(_)
                key.append(k)

            return val, key

        # make fig, gridspec
        fig = plt.figure(figsize=(fig_x / fig_dpi, fig_y / fig_dpi), dpi=fig_dpi)
        gs = GridSpec(1, 5)
        ss1 = gs.new_subplotspec((0, 0), rowspan=1, colspan=3)
        ss2 = gs.new_subplotspec((0, 3), rowspan=1, colspan=2)

        # ax1 fig
        ax1 = plt.subplot(ss1)
        ax1.grid(linewidth=0.2)
        ax1.set_axisbelow(True)

        xylim = 0.52 * self.res.inp.dia_prod
        ax1.set_xlim(-xylim, xylim)
        ax1.set_ylim(-xylim, xylim)

        # product
        product(self.res.inp.dia_prod)
        product(self.res.inp.dia_prod - 2.0 * self.res.inp.thk_prod, style="dashed", width=0.5)
        product(2.0 * self.res.inp.lim_incell, style="dashed", width=0.5)

        # slit
        slit()

        # cell
        incell()
        outcell()

        # table
        val, key = table()

        # ax2 table
        ax2 = plt.subplot(ss2)
        tab = ax2.table(cellText=val, rowLabels=key, loc="center", colWidths=[1, 1])
        for _, cell in tab.get_celld().items():
            cell.set_height(1 / len(val))
        ax2.axis("off")
        tab.set_fontsize(16)

        # save fig
        plt.tight_layout()
        fig.savefig(self.path[0] / f"index{self.id:0>4}.png", facecolor="w", dpi=fig_dpi)
        plt.close(fig)

    def data_to_series(self) -> pd.Series:
        """
        1水準分の設計変数, 評価変数をまとめる

        Returns
        -------
        res : pd.Series
            結果
        """
        tmp1 = pd.Series(self.res.inp.__dict__)
        tmp2 = pd.Series(self.table_b)
        tmp3 = pd.Series({"ERROR": "None"})
        _ = pd.concat([tmp1, tmp2], axis=0)
        res = pd.concat([_, tmp3], axis=0)
        return res
