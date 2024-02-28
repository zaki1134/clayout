# %%
import re

import sys
from pathlib import Path, WindowsPath

from datetime import datetime
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec


def main() -> None:
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        return None

    # Get the input file path from command-line arguments
    inp_path = sys.argv[1]

    # Check if the input file exists
    if not WindowsPath(inp_path).exists():
        return None

    # Create a directory path with the current timestamp if it doesn't exist
    dir_path = Path(datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    # main
    for row in pd.read_csv(inp_path).itertuples():
        # Calculate the limit radius based on row values
        limit_radius = 0.5 * row.dia_prod - row.thk_prod

        # Determine the shape based on row values
        if row.shape_incell == "circle":
            shape = (row.shape_incell, row.shape_incell, row.shape_outcell)
        elif row.shape_incell == "hexagon":
            shape = (row.shape_incell, "heptagon", row.shape_outcell)

        # Calculate initial coordinates based on row values
        init_df = calc_init_coordinates(
            shape,
            row.dia_prod,
            row.thk_i2o,
            row.pitch_x,
            row.pitch_y,
            row.ratio_slit,
        )

        # Copy coordinates based on row values
        copied_df = copy_coordinates(
            init_df,
            row.dia_prod,
            row.pitch_x,
            row.pitch_slit,
            row.ratio_slit,
        )

        # Adjust coordinates based on row mode
        if row.mode_cell:
            copied_df["x"] += 0.5 * row.pitch_x
        if row.mode_slit:
            copied_df["y"] += 0.5 * row.pitch_slit

        # Filter coordinates based on row values and calculated limit radius
        filtered_df = filter_coordinates(
            copied_df,
            row.dia_incell,
            row.thk_cc,
            row.thk_x1,
            row.thk_y1,
            row.lim_slit,
            limit_radius,
        )

        # Perform result post-processing
        post = ResultPost(row, filtered_df)
        post.draw()

        # Write result to file in the directory path
        # post.write_file(dir_path)


def calc_init_coordinates(
    shape: tuple,
    dia_prod: float,
    thk_i2o: float,
    pitch_x: float,
    pitch_y: float,
    ratio_slit: int,
) -> pd.DataFrame:
    """
    初期座標を計算します。

    Parameters:
        shape (tuple): 形状情報(str, str, str)
        dia_prod (float): dia_prod
        thk_i2o (float): thk_i2o
        pitch_x (float): pitch_x
        pitch_y (float): pitch_y
        ratio_slit (int): ratio_slit

    Returns:
        pd.DataFrame: 計算された初期座標データを含むDataFrame
    """
    # empty dataframe
    result = DataFrameEditor()

    # add dataframe(slit)
    result.add_coordinate("slit", "standard", 0.0, np.array([(0.0, 0.0)]))

    # initial outcell
    num = np.int64(np.ceil(dia_prod / pitch_x))
    x_coordinates = np.arange(-num * pitch_x, num * pitch_x, pitch_x)
    coordinate_oc = np.column_stack((x_coordinates, np.zeros_like(x_coordinates)))

    # add dataframe(outcell)
    result.add_coordinate("outcell", shape[2], 0.0, coordinate_oc)

    # initial incell(step1)
    coordinates_ic_step1 = np.copy(coordinate_oc)
    coordinates_ic_step1[:, 0] += 0.5 * pitch_x
    coordinates_ic_step1[:, 1] += thk_i2o

    # add dataframe(incell)
    if shape[0] == shape[1]:
        result.add_coordinate("incell", shape[0], 0.0, coordinates_ic_step1)
    else:
        result.add_coordinate("incell", shape[1], 0.0, coordinates_ic_step1)

    # initial incell(after step1)
    for i in range(1, ratio_slit):
        coordinates_ic = np.copy(coordinates_ic_step1)
        coordinates_ic[:, 0] += 0.5 * (i % 2) * pitch_x
        coordinates_ic[:, 1] += pitch_y * i

        # add dataframe(incell)
        if i == ratio_slit - 1:
            result.add_coordinate("incell", shape[1], 180.0, coordinates_ic)
        else:
            result.add_coordinate("incell", shape[0], 0.0, coordinates_ic)

    return result.info


def copy_coordinates(
    ref_df: pd.DataFrame,
    dia_prod: float,
    pitch_x: float,
    pitch_slit: float,
    ratio_slit: int,
) -> pd.DataFrame:
    """
    基準座標をコピーして新しいDataFrameを生成

    Parameters:
        ref_df (pd.DataFrame): 基準座標(DataFrame)
        dia_prod (float): dia_prod
        pitch_x (float): pitch_x
        pitch_slit (float): pitch_slit
        ratio_slit (int): ratio_slit

    Returns:
        pd.DataFrame: 新しい座標データを含むDataFrame
    """
    # calc slit positions
    num = np.int64(np.ceil(dia_prod / pitch_slit))
    y_coordinates = np.arange(-num * pitch_slit, num * pitch_slit, pitch_slit)
    coordinate_sl = np.column_stack((np.zeros_like(y_coordinates), y_coordinates))

    # offset switch
    switches = np.arange(-num, num + 1) % 2 != 0 if ratio_slit % 2 == 0 else np.zeros(2 * num + 1, dtype=bool)

    # calculate x offset
    x_offsets = np.where(switches, 0.5 * pitch_x, 0)

    # copy and concatenate DataFrame
    copied_dfs = []
    for y, x_offset in zip(coordinate_sl, x_offsets):
        copied_df = ref_df.copy()
        copied_df["y"] += y[1]
        copied_df["x"] += x_offset
        copied_dfs.append(copied_df)

    return pd.concat(copied_dfs, ignore_index=True)


def filter_coordinates(
    df: pd.DataFrame,
    dia_incell: float,
    thk_cc: float,
    thk_x1: float,
    thk_y1: float,
    limit_y: float,
    limit_r: float,
) -> pd.DataFrame:
    """
    座標をフィルタリングし、制限値内に収まらない座標を削除します。

    Parameters:
        df (pd.DataFrame): 座標データを含むDataFrame
        dia_incell (float): dia_incell
        thk_cc (float): thk_cc
        thk_x1 (float): thk_x1
        thk_y1 (float): 八角形寸法
        limit_y (float): Y軸方向の制限値
        limit_r (float): 半径の制限値

    Returns:
        pd.DataFrame: フィルタリングされた座標データを含むDataFrame
    """
    # Create a copy of the input DataFrame
    ref_df = df.copy()

    # Add a new column to mark coordinates for removal
    ref_df["remove"] = False

    # Iterate over unique shapes in the DataFrame
    for shape in ref_df["shape"].unique():
        if shape == "standard":
            condition1 = (ref_df["category"] == "slit") & (ref_df["shape"] == shape)
            condition2 = abs(ref_df["y"]) >= limit_y
            ref_df.loc[(condition1 & condition2), "remove"] = True

        elif shape == "circle":
            condition1 = (ref_df["category"] == "incell") & (ref_df["shape"] == shape)
            cell_coordinates = np.column_stack((ref_df["x"], ref_df["y"]))
            condition2 = np.linalg.norm(cell_coordinates, axis=1) >= limit_r - 0.5 * dia_incell
            ref_df.loc[(condition1 & condition2), "remove"] = True

        elif shape == "hexagon":
            condition1 = (ref_df["category"] == "incell") & (ref_df["shape"] == shape)
            polygon = vertex_hex(0.5 * dia_incell)
            ref_df.loc[condition1, "remove"] = ref_df[condition1].apply(
                lambda row: is_in_limit(limit_r, polygon, row["x"], row["y"], row["angle"]), axis=1
            )

        elif shape == "heptagon":
            condition1 = (ref_df["category"] == "incell") & (ref_df["shape"] == shape)
            polygon = vertex_hep(0.5 * dia_incell)
            ref_df.loc[condition1, "remove"] = ref_df[condition1].apply(
                lambda row: is_in_limit(limit_r, polygon, row["x"], row["y"], row["angle"]), axis=1
            )

        elif shape == "octagon":
            condition1 = (ref_df["category"] == "outcell") & (ref_df["shape"] == shape)
            polygon = vertex_oct(thk_cc, thk_x1, thk_y1)
            ref_df.loc[condition1, "remove"] = ref_df[condition1].apply(
                lambda row: is_in_limit(limit_r, polygon, row["x"], row["y"], row["angle"]), axis=1
            )
            condition2 = abs(ref_df["y"]) >= limit_y
            ref_df.loc[(condition1 & condition2), "remove"] = True

    # Remove coordinates marked for removal
    filtered_df = ref_df[ref_df["remove"] != True]
    filtered_df.drop(columns=["remove"], inplace=True)

    return filtered_df


def is_in_limit(
    limit: float,
    polygon: np.ndarray,
    offset_x: float,
    offset_y: float,
    angle: float = 0.0,
) -> bool:
    """
    多角形の頂点が与えられた制限値内に収まるかどうかを確認
    中心座標を回転+オフセットした多角形の各頂点のR座標を得る
    それら頂点の座標が1つでも制限値外にあればTrueを返す

    Parameters:
        limit (float): 制限値
        polygon (np.ndarray): 多角形の頂点座標.サイズ(N, 2)
        offset_x (float): X軸方向のオフセット
        offset_y (float): Y軸方向のオフセット
        angle (float, optional): 回転角度.デフォルト値は0.0

    Returns:
        bool: 多角形頂点が1つでも制限値外にある場合はTrue、それ以外の場合はFalse
    """
    # Make a copy of the input polygon array
    copied_polygon = polygon.copy()

    # Create a rotation matrix using the given angle
    radian = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]])

    # Rotate the copied polygon points using the rotation matrix
    rot_polygon = np.dot(copied_polygon, rot_mat)

    # Offset the rotated polygon by the specified offsets
    rot_polygon[:, 0] += offset_x
    rot_polygon[:, 1] += offset_y

    # Check if any point in the rotated polygon exceeds the limit
    check = np.linalg.norm(rot_polygon, axis=1) >= limit

    # Return True if any point exceeds the limit, otherwise False
    return np.any(check)


def vertex_slit(radius_prod: float, thk_slit: float, y: float) -> np.ndarray:
    """
    スリット形状を多角形で表現.その頂点の座標を計算

    Parameters:
        radius_prod (float): 製品半径
        thk_slit (float): スリット厚
        y (float): スリットのY方向座標

    Returns:
        np.ndarray: 多角形の頂点座標.サイズ(N, 2)
    """
    # Calculate the y-coordinates of the reference points for the slit
    ref_points_y = np.array([y - 0.5 * thk_slit, y + 0.5 * thk_slit])

    # Calculate the x-coordinates of the reference points
    ref_points_x = radius_prod * np.cos(np.arcsin(ref_points_y / radius_prod))

    # Calculate the angles corresponding to the reference points
    ref_angles = np.arctan2(ref_points_y, ref_points_x)

    # Generate a set of angles evenly spaced between the minimum and maximum reference angles
    angles = np.linspace(np.min(ref_angles), np.max(ref_angles), 10)

    # Calculate the coordinates of points on the arc using the generated angles
    coordinates = radius_prod * np.column_stack((np.cos(angles), np.sin(angles)))

    # Duplicate the coordinates by reflecting them across the y-axis
    copied_coordinates = np.copy(coordinates)
    copied_coordinates[:, 0] *= -1.0
    coordinates = np.vstack([coordinates, copied_coordinates])

    # Sort the coordinates based on the angle they make with the origin
    sorted_angles = np.arctan2(coordinates[:, 1], coordinates[:, 0])
    sorted_angles[sorted_angles < 0] += 2 * np.pi
    slit_points = coordinates[np.argsort(sorted_angles)]

    return slit_points


def vertex_hex(radius_incircle: float) -> np.ndarray:
    """
    正六角形の頂点の座標を計算

    Args:
        radius_incircle (float): 内接円半径

    Returns:
        np.ndarray: 正六角形の頂点の座標.サイズ(N, 2)
    """
    # Calculate the radius of the circumscribed circle
    circumscribed_radius = radius_incircle / np.cos(np.pi / 6)

    # List of angles
    angles = np.arange(-np.pi / 6, 3 * np.pi / 2, np.pi / 3)

    # Calculate coordinates of 6 points
    hex_points = circumscribed_radius * np.column_stack((np.cos(angles), np.sin(angles)))

    return hex_points


def vertex_hep(radius_incircle: float, rot_angle: float = 0.0) -> np.ndarray:
    """
    七角形の頂点の座標を計算.正六角形に1辺追加

    Args:
        radius_incircle (float): 元になる正六角形の内接円半径
        rot_angle (float, optional): 原点中心からの回転角度(degree).デフォルト値は0.0

    Returns:
        np.ndarray: 七角形の頂点の座標.サイズ(N, 2)
    """
    # Calculate the radius of the circumscribed circle
    circumscribed_radius = radius_incircle / np.cos(np.pi / 6)

    # Calculate coordinates of 5 points
    angles_5_points = np.arange(-np.pi / 6, 4 * np.pi / 3, np.pi / 3)
    hep_points = circumscribed_radius * np.column_stack((np.cos(angles_5_points), np.sin(angles_5_points)))

    # Calculate coordinates of 2 points
    angles_2_points = np.array([4 * np.pi / 3, 5 * np.pi / 3])
    dummy_points = radius_incircle * np.column_stack((np.cos(angles_2_points), np.sin(angles_2_points)))

    # Stack all points together
    hep_points = np.vstack([hep_points, dummy_points])

    # Calculate coordinates of rotated 7 points
    radian = np.deg2rad(rot_angle)
    rot_mat = np.array([[np.cos(radian), -np.sin(radian)], [np.sin(radian), np.cos(radian)]])
    hep_points = np.dot(hep_points, rot_mat)

    return hep_points


def vertex_oct(thk_cc: float, thk_x1: float, thk_y1: float) -> np.ndarray:
    """
    八角形の頂点の座標を計算

    Parameters:
        thk_cc (float): 八角形寸法
        thk_x1 (float): 八角形寸法
        thk_y1 (float): 八角形寸法

    Returns:
        np.ndarray: 八角形の頂点の座標.サイズ(N, 2)
    """
    # Calculate coordinates of 4 points
    oct_points = np.array(
        [
            (thk_x1 + thk_cc, thk_y1),
            (thk_x1, thk_y1 + thk_cc),
            (-thk_x1, thk_y1 + thk_cc),
            (-(thk_x1 + thk_cc), thk_y1),
        ]
    )

    # Calculate coordinates of rotated 4 points
    rot_mat = np.array([[np.cos(np.pi), -np.sin(np.pi)], [np.sin(np.pi), np.cos(np.pi)]])
    dummy_points = np.dot(oct_points, rot_mat)

    # Stack all points together
    oct_points = np.vstack([oct_points, dummy_points])

    return oct_points


class DataFrameEditor:
    """
    pandas.DataFrameを編集するためのクラス

    Attributes:
        column_types (dict): 各列名とデータ型を指定する辞書
    """

    column_types = {
        "category": np.str_,
        "shape": np.str_,
        "angle": np.float64,
        "x": np.float64,
        "y": np.float64,
    }

    def __init__(self) -> None:
        """
        指定された列名とデータ型で空のDataFrameを初期化、インスタンス変数に格納
        """
        # Initialize an empty DataFrame with specified column names
        empty_df = pd.DataFrame(columns=list(self.column_types.keys()))

        # Convert the data types of the DataFrame columns as per the defined column_types dictionary
        self.info = empty_df.astype(self.column_types)

    def add_coordinate(self, category: str, shape: str, angle: float, coordinates: np.ndarray) -> None:
        """
        インスタンス変数のDataFrameに情報を追加

        Args:
            category (str): データのカテゴリ
            shape (str): 形状
            angle (float): セルを傾ける角度(degree)
            coordinates (np.ndarray): N個の中心座標.サイズ(N, 2)
        """
        # Skip if coordinates array is empty
        if not coordinates.size:
            return

        # Create DataFrame from XY coordinates
        coordinates_df = pd.DataFrame(coordinates, columns=["x", "y"])

        # Add object category, shape, and angle to each row
        info_df = pd.DataFrame(
            {
                "category": [category] * len(coordinates),
                "shape": [shape] * len(coordinates),
                "angle": [angle] * len(coordinates),
            }
        )

        # Concatenate coordinates DataFrame with info DataFrame
        combined_df = pd.concat([info_df, coordinates_df], axis=1)

        # Append the new DataFrame to the existing DataFrame
        self.info = pd.concat([self.info, combined_df], ignore_index=True)


class ResultPost:
    """
    結果を描画,保存するクラス

    Attributes:
        info (pd.DataFrame):結果データ(DataFrame)
        inp (tuple):入力データ(tuple)
    """

    def __init__(self, inp: tuple, df: pd.DataFrame) -> None:
        self.info = df
        self.inp = inp

    def write_file(self, path: WindowsPath) -> None:
        """
        以下の3ファイルを出力
        0000_layout.csv : categoly, x, yを記載
        0000_parameters.csv : 入力値
        0000.png : 画像出力

        Args:
            path (WindowsPath): 結果保存先パス
        """
        # Generate an index with leading zeros for formatting
        idx = f"{self.inp.Index:04}"
        dir_path = path / idx

        # Write the layout information to a CSV file
        condition = ["category", "x", "y"]
        self.info[condition].to_csv(dir_path / (idx + "_layout.csv"), float_format="%.5e", index=False)

        # Write the input parameters to a CSV file
        inp_df = pd.DataFrame([self.inp])
        inp_df.to_csv(dir_path / (idx + "_parameters.csv"), float_format="%.5e", index=False)

        # Save the plot as a PNG file
        plt.savefig(dir_path / (idx + ".png"))

    def draw(self) -> None:
        """
        画像出力
        """

        def pre(size: float, fig_x: int = 1920, fig_y: int = 1080, fig_dpi: int = 100) -> Tuple[plt.Axes, plt.Axes]:
            """
            指定されたサイズと解像度で図を作成

            Args:
                size (float): 図のサイズを指定
                fig_x (int, optional): 図の幅(pixel)のデフォルト値は1920
                fig_y (int, optional): 図の高さ(pixel)のデフォルト値は1080
                fig_dpi (int, optional): 図のDPIのデフォルト値は100

            Returns:
                Tuple[plt.Axes, plt.Axes]: 2つのAxesオブジェクトを含むタプル
                    1つ目は描画用, 2つ目は入力値テーブル用
            """
            # Create a new figure with specified dimensions and DPI
            fig = plt.figure(figsize=(fig_x / fig_dpi, fig_y / fig_dpi), dpi=fig_dpi)
            fig.subplots_adjust(left=0.025, right=0.99, bottom=0.025, top=0.99)

            # Define a grid specification with one row and ten columns
            gs = GridSpec(1, 10)
            ss1 = gs.new_subplotspec((0, 0), colspan=7)
            ss2 = gs.new_subplotspec((0, 8), colspan=2)

            # Create the first subplot (ax1) within the specified subplotspec
            ax1 = plt.subplot(ss1)
            ax1.grid(linewidth=0.2)
            ax1.set_axisbelow(True)
            ax1.set_aspect("equal", adjustable="datalim")

            # Set the display range for ax1
            scale = 0.52 * size
            ax1.set_xlim(-scale, scale)
            ax1.set_ylim(-scale, scale)

            # Create the second subplot (ax2) within the specified subplotspec
            ax2 = plt.subplot(ss2)
            ax2.axis("off")

            return ax1, ax2

        def draw_circle(
            ax: plt.Axes,
            diameter: float,
            facecolor: str = "#BFBFBF",
            transparency: float = 1.0,
            edgecolor: str = "black",
            linestyle: str = "solid",
            offset_x: float = 0.0,
            offset_y: float = 0.0,
        ) -> None:
            """
            円を描画

            Args:
                ax (plt.Axes): 描画先のAxes
                diameter (float): 円の直径
                facecolor (str, optional): 塗りつぶし色.デフォルト値は"#BFBFBF"
                transparency (float, optional): 透明度.デフォルト値は1.0
                edgecolor (str, optional): 境界線の色.デフォルト値は"black"
                linestyle (str, optional): 境界線のスタイル.デフォルト値は"solid"
                x (float, optional): 円の中心x座標.デフォルト値は0.0
                y (float, optional): 円の中心y座標.デフォルト値は0.0
            """
            circles = patches.Circle(
                xy=(offset_x, offset_y),
                radius=0.5 * diameter,
                facecolor=facecolor,
                alpha=transparency,
                edgecolor=edgecolor,
                linestyle=linestyle,
            )
            ax.add_patch(circles)

        def draw_polygon(
            ax: plt.Axes,
            ref_coordinate: np.ndarray,
            offset_x: float = 0.0,
            offset_y: float = 0.0,
            facecolor: str = "#4F81BD",
        ) -> None:
            """
            多角形を描画

            Args:
                ax (plt.Axes): 描画先のAxes
                ref_coordinate (np.ndarray): 多角形の各点の座標.サイズ(N, 2)
                offset_x (float, optional): 中心x座標.デフォルト値は0.0
                offset_y (float, optional): 中心y座標.デフォルト値は0.0
                facecolor (str, optional): 塗りつぶし色.デフォルト値は"#4F81BD"
            """
            points = np.copy(ref_coordinate)
            points[:, 0] += offset_x
            points[:, 1] += offset_y
            polygons = plt.Polygon(points, closed=True, facecolor=facecolor, fill=True, linewidth=0)
            ax.add_patch(polygons)

        def table(ax: plt.Axes, inp: tuple, info: pd.DataFrame) -> None:
            """
            テーブルを描画.出力パラメータはtarget(tuple)で指定

            Args:
                ax (plt.Axes): 描画先のAxes
                inp (tuple): 入力データ(tuple)
                info (pd.DataFrame): 結果データ(DataFrame)
            """
            # Select parameters to export
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
            inp_dict = inp._asdict()
            export_params = {key: inp_dict[key] for key in target}

            # Extract data for different categories
            df_incell = info[info["category"] == "incell"]
            df_outcell = info[info["category"] == "outcell"]
            df_slit = info[info["category"] == "slit"]

            # Calculate additional output parameters
            thk_mem = inp_dict["thk_top"] + inp_dict["thk_mid"] + inp_dict["thk_bot"]
            diameter_effective = inp_dict["dia_incell"] - 2.0 * thk_mem
            area_incell = 0.25 * (np.pi * diameter_effective**2.0)
            area_prod = 0.25 * (np.pi * inp_dict["dia_prod"] ** 2.0)
            area_mem = np.pi * diameter_effective * inp_dict["ln_prod"] * len(df_incell)
            total_area_incell = area_incell * len(df_incell)

            export_params["N(incell)"] = len(df_incell)
            export_params["N(outcell)"] = len(df_outcell)
            export_params["N(slit)"] = len(df_slit)
            export_params["A(membrane)"] = area_mem
            export_params["A(incell)"] = total_area_incell
            export_params["R_A(incell/prod)"] = (total_area_incell / area_prod) * 100.0

            # Format the output parameters
            res = {}
            for k, v in export_params.items():
                if re.compile(r"^dia_").match(k) or re.compile(r"^thk_").match(k) or re.compile(r"^ln_").match(k):
                    res[k] = f"{v:.2f} [mm]"
                elif re.compile(r"^A\(").match(k):
                    res[k] = f"{v:.2e} [mm2]"
                elif re.compile(r"^R_A\(").match(k):
                    res[k] = f"{v:.1f} [%]"
                else:
                    res[k] = f"{v}"

            # Create table
            key = [k for k in res.keys()]
            val = [[v] for v in res.values()]
            tbl = ax.table(cellText=val, rowLabels=key, loc="center")
            tbl.set_fontsize(16)
            for _, cell in tbl.get_celld().items():
                cell.set_height(1 / len(val))

        # figure, Axes
        # ax1, ax2 = pre(self.inp.dia_prod)
        ax1, ax2 = pre(self.inp.pitch_slit)

        # product(ax1)
        dia = self.inp.dia_prod
        draw_circle(ax1, dia, facecolor="None")
        draw_circle(ax1, dia, transparency=0.5)
        dia = self.inp.dia_prod - 2.0 * self.inp.thk_prod
        draw_circle(ax1, dia, facecolor="None", transparency=1, linestyle="dashed")

        # slit(ax1)
        coordinate = self.info[self.info["category"] == "slit"][["x", "y"]].to_numpy()
        for _, y in coordinate:
            slit_points = vertex_slit(0.5 * self.inp.dia_prod, self.inp.thk_slit, y)
            draw_polygon(ax1, slit_points, "#97B6D8")

        # incell(ax1)
        if self.inp.shape_incell == "circle":
            coordinate = self.info[self.info["category"] == "incell"][["x", "y"]].to_numpy()
            dia = self.inp.dia_incell - 2.0 * (self.inp.thk_top + self.inp.thk_mid + self.inp.thk_bot)
            [draw_circle(ax1, diameter=dia, facecolor="#4F81BD", edgecolor="None", x=x, y=y) for x, y in coordinate]

        elif self.inp.shape_incell == "hexagon":
            condition = (self.info["category"] == "incell") & (self.info["shape"] == "hexagon")
            hex_coordinates = self.info[condition][["x", "y"]].to_numpy()
            hex_points = vertex_hex(0.5 * self.inp.dia_incell)
            [draw_polygon(ax1, hex_points, x, y) for x, y in hex_coordinates]

            condition = (self.info["category"] == "incell") & (self.info["shape"] == "heptagon")
            hep_info = self.info[condition][["angle", "x", "y"]].to_numpy()
            for angle, x, y in hep_info:
                hep_points = vertex_hep(0.5 * self.inp.dia_incell, angle)
                draw_polygon(ax1, hep_points, x, y)

        # outcell(ax1)
        coordinate = self.info[self.info["category"] == "outcell"][["x", "y"]].to_numpy()
        oct_points = vertex_oct(self.inp.thk_cc, self.inp.thk_x1, self.inp.thk_y1)
        [draw_polygon(ax1, oct_points, x, y, "#76913C") for x, y in coordinate]

        # table(ax2)
        table(ax2, self.inp, self.info)


if __name__ == "__main__":
    main()

# %%
