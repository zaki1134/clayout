# my_tools

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib import gridspec


def draw_cell_layout(cnt: int, pos_cell: list, pos_slit: list, df, path: str,) -> None:
    def draw_circle(axes, radius, center=(0.0, 0.0), color="black", linestyle="solid") -> None:
        """
        draw circle

        Parameters
        ----------
        axes : axes
            axes
        radius : float
            radius
        center : tuple, optional
            circle center, by default (0.0, 0.0)
        color : str, optional
            line color, by default "black"
        linestyle : str, optional
            line style, by default "solid"
        """
        theta = np.linspace(0.0, 2.0 * np.pi, 360.0)

        xxx = center[0] + radius * np.cos(theta)
        yyy = center[1] + radius * np.sin(theta)

        axes.plot(xxx, yyy, color=color, linestyle=linestyle)

        return None

    def draw_slit(axes, pos_slit: list, slit_width: float, diameter_product: float, dbg: bool = True) -> None:
        """
        draw slit

        Parameters
        ----------
        axes : axes
            axes
        pos_slit : list
            slit positions
        slit_width : float
            input value
        diameter_product : float
            input value
        dbg : bool, optional
            debug, by default True

        Returns
        -------
        _type_
            _description_
        """
        for _ in pos_slit:
            y_t = [_ + 0.5 * slit_width] * 2
            y_b = [_ - 0.5 * slit_width] * 2

            if dbg:
                x_t = [
                    -np.sqrt((0.5 * diameter_product) ** 2.0 - y_t[0] ** 2.0),
                    np.sqrt((0.5 * diameter_product) ** 2.0 - y_t[1] ** 2.0),
                ]
                x_b = [
                    -np.sqrt((0.5 * diameter_product) ** 2.0 - y_b[0] ** 2.0),
                    np.sqrt((0.5 * diameter_product) ** 2.0 - y_b[1] ** 2.0),
                ]

            else:
                x_t = [-diameter_product, diameter_product]
                x_b = [-diameter_product, diameter_product]

            axes.plot(x_t, y_t, color="blue", linewidth=1, linestyle="dashed")
            axes.plot(x_b, y_b, color="blue", linewidth=1, linestyle="dashed")

        return None

    # make fig, axes
    fig = plt.figure(figsize=(8, 5))
    spec = gridspec.GridSpec(ncol=2, nrow=1, width_ratios=[2, 1])

    axes = fig.add_subplot(spec[0])
    axes.grid(linewidth=0.2)
    axes.set_axisbelow(True)
    aaa = 0.5 * df.loc[cnt, "diameter_product"] + 10
    axes.set_xlim(-aaa, aaa)
    axes.set_ylim(-aaa, aaa)

    # draw product
    draw_circle(axes, radius=0.5 * df.loc[cnt, "diameter_product"])
    draw_circle(
        axes, radius=(0.5 * df.loc[cnt, "diameter_product"] - df.loc[cnt, "thickness_product"]), line_style="dashed"
    )

    # draw slit
    draw_slit(axes, pos_slit, df.loc[cnt, "thickness_slit"], df.loc[cnt, "diameter_product"], False)

    # draw cell
    for _ in pos_cell:
        r_cell = 0.5 * df.loc[cnt, "diameter_cell"] - (
            df.loc[cnt, "thickness_bot"] + df.loc[cnt, "thickness_mid"] + df.loc[cnt, "thickness_top"]
        )
        axes.add_patch(patch.Circle(xy=_, radius=r_cell))

    # draw table
    temp_str = np.empty(0)
    for _ in df.columns:
        if _ == "membrane_area" or _ == "cell_area" or _ == "ratio_area":
            aaa = "{:.4e}".format(df.loc[cnt, _])
            temp_str = np.appned(temp_str, aaa)

        else:
            temp_str = np.append(temp_str, str(df.loc[cnt, _]))

    ay = fig.add_subplot(spec[1])
    ay.axis("off")
    ay.table(cellText=temp_str.reshape(-1, 1), rowLabels=df.columns, loc="center", bbox=[0.6, 0, 1, 1])

    # save fig
    fig.savefig(path, facecolor="w", bbox_inches="tight", dpi=300)
    plt.close(fig)

    return None


def draw_wo_table(
    c_pos: list,
    s_pos: list,
    d_prod: float,
    t_prod: float,
    t_slit: float,
    d_cell: float,
    t_bot: float,
    t_mid: float,
    t_top: float,
):
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
    pass
