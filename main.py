# %%

from cell_layout_v4 import *


def set_parameters():
    """
    unit mm
    リスト内の変数型を統一

    Returns
    -------
    res : dict
        input parameters
    """
    res = {
        "dia_incell": [2.0],
        "dia_prod": [30.0],
        "thk_bot": [100.0e-3],
        "thk_mid": [20.0e-3],
        "thk_top": [1.0e-3],
        "thk_wall_outcell": [0.2, -0.4],  # < 0.0 -> thk_wall
        "thk_outcell": [3.0],
        "thk_prod": [5.0],
        "thk_wall": [0.4],
        "thk_c2s": [0.0],  # < 0.0 -> thk_wall
        "thk_slit": [1.0],
        "ln_prod": [1000.0],
        "ln_slit": [30.0],
        "ln_edge": [20.0],
        "ln_glass_seal": [15.0],
        "ratio_slit": [4],  # int
        "mode_cell": [True, False],  # bool
        "mode_slit": [True, False],  # bool
    }
    return res


def main():
    # make directory
    path = make_dir()
    if path is not None:
        pass
    else:
        return None

    # parameters
    param = set_parameters()
    inp = dict_to_df(param)

    # main
    stack = pd.DataFrame()
    for num in range(len(inp)):
        try:
            case = CirOct(**inp.loc[num, :])
        except (TypeError, ValueError) as e:
            case, res, post = None, None, None
            tmp = except_case(e)
        else:
            res = Result(case)
            post = Post(res, path, num)
            post.draw()
            tmp = post.data_to_series()

        stack = pd.concat([stack, tmp], axis=1)
        del (case, res, post)

    # write csv
    stack = stack.T.reset_index(drop=True)
    stack.to_csv(path[0] / "data.csv")


if __name__ == "__main__":
    main()

# %%
