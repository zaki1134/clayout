# %%
import pathlib
import datetime
import itertools
import math
from typing import Any, Tuple

from tqdm import tqdm
import pandas as pd


def main() -> None:
    # dir_path = pathlib.Path(datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    # if not dir_path.exists():
    #     dir_path.mkdir()

    # setting parameters
    df = set_parameters()
    df["num_cell"] = -1
    df["num_slit"] = -1
    df["membrane_area"] = -1.0
    df["cell_area"] = -1.0
    df["ratio_area"] = -1.0

    for num in tqdm(range(len(df))):
        # make obj
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

        # calc slit positions
        slit_positions = list(set_pos_slit(case.p_slit, case.lim_slit, case.m_slit))

        # calc cell positions
        unit_cell_positions = list(
            set_pos_cell_unit(case.p_cell, case.d_prod, case.unit_y_cell, case.r_slit, case.m_cell)
        )
        copy_cell_positions = list(set_pos_cell_copy(unit_cell_positions, slit_positions, case.os_cell_slit))
        cell_positions = list(set_pos_cell_filter(copy_cell_positions, 0.5 * case.d_prod - case.t_prod, case.d_cell))

        # calc value
        num_cell = calc_num_cell(cell_positions, case.m_slit)
        df.loc[num, "num_cell"] = num_cell

        df.loc[num, "num_slit"] = len(slit_positions)

        effective_diameter_cell = case.d_cell - 2.0 * (case.t_bot + case.t_mid + case.t_top)
        membrane_area = calc_membrane_area(num_cell, case.length, effective_diameter_cell)
        df.loc[num, "membrane_area"] = membrane_area

        cell_area, ratio_area = calc_ratio_area(num_cell, effective_diameter_cell, case.d_prod)
        df.loc[num, "cell_area"] = cell_area
        df.loc[num, "ratio_area"] = ratio_area

        # draw
        draw(
            cell_positions,
            slit_positions,
            case.d_prod,
            case.t_prod,
            case.t_slit,
            case.d_cell,
            case.t_bot,
            case.t_mid,
            case.t_top,
        )

    # export conditions
    # df.to_csv(str(dir_path) + "\\data.csv")

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
        self.unit_y_cell = self.p_cell * math.sin(math.pi / 3.0)
        self.p_slit = self.t_slit + (self.r_slit - 1) * self.unit_y_cell + self.d_cell + 2.0 * self.t_rib
        self.lim_slit = 0.5 * self.d_prod - self.t_prod - self.d_cell - self.t_rib - 0.5 * self.t_slit
        self.os_cell_slit = 0.5 * self.t_slit + self.t_rib + 0.5 * self.d_cell


def set_parameters() -> Any:
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
    ]

    result = pd.DataFrame(param_set, columns=title)

    return result


def set_pos_slit(s_pitch: float, s_limit: float, s_mode: str) -> list:
    result = []
    i = 0
    if s_mode == "A":
        while True:
            pitch = s_pitch * i
            if pitch > s_limit:
                break

            result.append(pitch)
            i += 1

    else:
        while True:
            pitch = s_pitch * (i - 0.5)
            if pitch > s_limit:
                break

            result.append(pitch)
            i += 1

    return result


def set_pos_cell_unit(c_pitch: float, c_limit: float, c_pitch_y: float, r_slit: int, c_mode: str) -> list:
    result = []
    if c_mode == "A":
        for j in range(r_slit):
            i = -1
            while True:
                if c_pitch * i >= c_limit:
                    break

                if j % 2 == 0:
                    result.append([c_pitch * i, c_pitch_y * j])

                else:
                    result.append([c_pitch * (i + 0.5), c_pitch_y * j])

                i += 1

    else:
        for j in range(r_slit):
            i = -1
            while True:
                if c_pitch * i >= c_limit:
                    break

                if j % 2 == 0:
                    result.append([c_pitch * (i + 0.5), c_pitch_y * j])

                else:
                    result.append([c_pitch * i, c_pitch_y * j])

                i += 1

    return result


def set_pos_cell_copy(c_unit: list, s_pos: list, os: float) -> list:
    result = []

    for yyy in s_pos:
        temp = [[pos[0], pos[1] + yyy + os] for pos in c_unit]
        result.extend(temp)

    return result


def set_pos_cell_filter(c_copy: list, eff_radius: float, d_cell: float) -> list:
    result = [
        [pos[0], pos[1]]
        for pos in c_copy
        if (math.sqrt(pos[0] ** 2.0 + pos[1] ** 2.0) <= eff_radius - 0.5 * d_cell)
        and (pos[0] >= 0.0)
        and (pos[1] >= 0.0)
    ]

    return result


def calc_num_cell(c_pos: list, s_mode: str) -> int:
    if s_mode == "A":
        cnt = [_ for _ in c_pos if (_[0] != 0.0) and (_[1] != 0.0)]
        result = 4 * len(cnt) + 2 * (len(c_pos) - len(cnt))

    else:
        cnt = [_ for _ in c_pos if (_[0] != 0.0) and (_[1] != 0.0)]
        if c_pos[0][0] == 0.0 and c_pos[0][1] == 0.0:
            result = 4 * len(cnt) + 2 * (len(c_pos) - len(cnt) - 1) + 1

        else:
            result = 4 * len(cnt) + 2 * (len(c_pos) - len(cnt))

    return result


def calc_membrane_area(c_num: int, length: float, eff_d_cell: float) -> float:
    result = math.pi * eff_d_cell * length * c_num
    return result


def calc_ratio_area(c_num: int, eff_d_cell: float, d_prod: float) -> Tuple[float, float]:
    c_area = 0.25 * math.pi * (eff_d_cell ** 2.0) * c_num
    r_area = c_area / (0.25 * math.pi * (d_prod ** 2.0))

    return c_area, r_area


def draw(
    c_pos: list,
    s_pos: list,
    d_prod: float,
    t_prod: float,
    t_slit: float,
    d_cell: float,
    t_bot: float,
    t_mid: float,
    t_top: float,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patch

    def draw_arc(axes, radius, color="black", line_style="solid"):
        axes.add_patch(
            patch.Arc(
                xy=(0.0, 0.0),
                width=radius * 2,
                height=radius * 2,
                theta1=0,
                theta2=360,
                linewidth=1,
                edgecolor=color,
                linestyle=line_style,
            )
        )

        return None

    def draw_slit(axes, slit_center, slit_width, outer_diameter):
        for pos in slit_center:
            y_top = [pos + 0.5 * slit_width] * 2
            x_top = [
                -math.sqrt((0.5 * outer_diameter) ** 2.0 - y_top[1] ** 2.0),
                math.sqrt((0.5 * outer_diameter) ** 2.0 - y_top[1] ** 2.0),
            ]

            y_bot = [pos - 0.5 * slit_width] * 2
            x_bot = [
                -math.sqrt((0.5 * outer_diameter) ** 2.0 - y_bot[1] ** 2.0),
                math.sqrt((0.5 * outer_diameter) ** 2.0 - y_bot[1] ** 2.0),
            ]

            axes.plot(x_top, y_top, color="blue", linewidth=1, linestyle="dashed")
            axes.plot(x_bot, y_bot, color="blue", linewidth=1, linestyle="dashed")

        return None

    def draw_cell(axes, center, radius, color="black"):
        for pos in center:
            axes.add_patch(patch.Circle(xy=pos, radius=0.5 * radius, facecolor=color, edgecolor=color,))

        return None

    # set fig, ax
    fig = plt.figure(figsize=(10, 10), dpi=300)
    ax = fig.add_subplot(111)

    ax.grid(linewidth=0.2)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.55 * d_prod, 0.55 * d_prod)
    ax.set_ylim(-0.55 * d_prod, 0.55 * d_prod)

    # draw product
    draw_arc(ax, radius=0.5 * d_prod)
    draw_arc(ax, radius=0.5 * d_prod - t_prod, line_style="dashed")

    # draw slit
    draw_slit(ax, s_pos, t_slit, d_prod)

    # draw cell
    draw_cell(ax, center=c_pos, radius=d_cell - 2.0 * (t_bot + t_mid + t_top), color="blue")

    plt.show()

    return None


if __name__ == "__main__":
    main()
    print("****** Done ******")

# %%
