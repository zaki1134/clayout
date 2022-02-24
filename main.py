import numpy as np
import json

import itertools
import pandas as pd

import pathlib
import datetime
import my_tools


def main():
    # make dir
    dir_path = pathlib.Path(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not dir_path.exists():
        dir_path.mkdir()

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
        )

        # set base slit positions
        positions_bslit = set_pos_slit(case.p_slit, case.d_prod, case.m_slit)

        # set base cell positions
        positions_bcell = set_pos_cell(
            case.p_cell,
            0.6 * case.d_prod,
            case.y_cell_slit,
            case.y_cell_cell,
            case.r_slit,
            positions_bslit,
            case.m_cell,
        )

        # select slit
        positions_slit = select_slit(positions_bslit, case.lim_slit)

        # select cell
        positions_cell = select_cell(positions_bcell, case.d_prod, case.t_prod, case.d_cell)

        # calc value
        effective_diameter_cell = case.d_cell - 2.0 * (case.t_bot + case.t_mid + case.t_top)
        num_cell = len(positions_cell)
        df.loc[num, "num_cell"] = num_cell
        df.loc[num, "num_slit"] = len(positions_slit)
        df.loc[num, "membrane_area"] = np.pi * effective_diameter_cell * case.length + num_cell

        cell_area = 0.25 * np.pi * (effective_diameter_cell ** 2.0) * num_cell
        prod_area = cell_area / 0.25 * np.pi * (df.loc[num, "diameter_product"] ** 2.0) * num_cell
        df.loc[num, "cell_area"] = cell_area
        df.loc[num, "ratio_area"] = cell_area / prod_area

        # data output
        cnt_str = f"index{num:0>4}"

        my_tools.draw_cell_layout(
            num, positions_bcell, positions_bslit, pd.DataFrame(df.iloc[num, :].T, dir_path / cnt_str)
        )
        df.to_csv(dir_path / "data.csv")

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

        self.p_cell = self.d_cell + self.t_rib
        self.y_cell_cell = self.p_cell * np.sin(np.pi / 3.0)
        self.p_slit = self.t_slit + (self.r_slit - 1) * self.y_cell_cell + self.d_cell + 2.0 * self.t_rib
        self.lim_slit = 0.5 * self.d_prod - self.t_prod - self.d_cell - self.t_rib - 0.5 * self.t_slit
        self.y_cell_slit = 0.5 * self.t_slit + self.t_rib + 0.5 * self.d_cell


def set_parameters():
    # unit : mm
    diameter_cell = [2.0]
    diameter_product = [30.0]
    thickness_rib = [0.4]
    thickness_slit = [1.0]
    thickness_product = [1.0]
    thickness_bot = [250e-3]
    thickness_mid = [20e-3]
    thickness_top = [1e-3]
    length_product = [150.0]
    ratio_slit = [3]
    mode_cell = ["A"]
    mode_slit = ["A"]

    # calc value
    num_cell = [-1]
    num_slit = [-1]
    membrane_area = [-1.0]
    cell_area = [-1.0]
    ratio_area = [-1.0]

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
            num_cell,
            num_slit,
            membrane_area,
            cell_area,
            ratio_area,
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
        "num_cell",
        "num_slit",
        "membrane_area",
        "cell_area",
        "ratio_area",
    ]

    return pd.DataFrame(param_set, columns=title)


def set_pos_slit(s_pitch: float, s_limit: float, s_mode: str):
    """
    create more slits

    Parameters
    ----------
    s_pitch : float
        slit pitch
    s_limit : float
        slit limit
    s_mode : str
        mode_slit
    """
    result = []
    i = 0
    while True:
        if s_mode == "A":
            aaa = s_pitch * i

        else:
            aaa = s_pitch * (i + 0.5)

        if aaa >= s_limit:
            break

        result.append(aaa)

    bbb = [xxx for xxx in result if xxx > 0.0]

    for _ in bbb:
        result.append(-_)

    return np.array(result)


def set_pos_cell(
    c_pitch: float, c_limit: float, cs_pitch: float, cc_pitch: float, r_slit: int, pos_slit: list, c_mode: str
):
    result = []
    if c_mode == "A":
        for j in range(r_slit):
            i = -1
            while True:
                if c_pitch * i >= c_limit:
                    break

                if j % 2 == 0:
                    result.append([c_pitch * i, cs_pitch * j])

                else:
                    result.append([c_pitch * (i + 0.5), cs_pitch * j])

                i += 1

    else:
        for j in range(r_slit):
            i = -1
            while True:
                if c_pitch * i >= c_limit:
                    break

                if j % 2 == 0:
                    result.append([c_pitch * (i + 0.5), cs_pitch * j])

                else:
                    result.append([c_pitch * i, cs_pitch * j])

                i += 1

    return result


def select_slit(bslit: list, s_limit: float):
    """
    Select a slit other than the one beyond the limit.

    Parameters
    ----------
    bslit : list
        positions_bslit
    s_limit : float
        slit limit
    """
    return np.array([xxx for xxx in bslit if abs(xxx) < s_limit])


def select_cell(bcell: list, d_prod: float, t_prod: float, d_cell: float):
    """
    Select a cell other than the one beyond the limit.

    Parameters
    ----------
    bcell : list
        positions_bcell
    d_prod : float
        diameter_product
    t_prod : float
        thickness_product
    d_cell : float
        diameter_cell
    """
    result = []
    c_limit = 0.5 * d_prod - t_prod - 0.5 * d_cell
    for xy in bcell:
        radius = np.sqrt(xy[0] ** 2.0 + xy[1] ** 2.0)
        if radius < c_limit:
            radius.append(xy[0], xy[1])

    return np.array(result)


def dump_json(aaa, fname="test_json", sw_sort: bool = False):
    with open(fname, "wt") as f:
        json.dump(aaa, f, sort_keys=sw_sort, index=4)

    return None


if __name__ == "__main__":
    main()
    print("****** Done ******")
