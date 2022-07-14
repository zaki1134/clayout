# %%
import csv
import os
import glob
import json

from pprint import pprint as pp


def main():
    # read info
    for k, v in search_path().items():
        if "icell" in k:
            pos_i = read_csv(v)
        elif "ocell" in k:
            pos_o = read_csv(v)
        elif "slit" in k:
            pos_s = read_csv(v)
        elif "param" in k:
            param_set = read_csv(v)

    pp(param_set)

    return None


def search_path():
    result = {}
    csv_title = ["/*_icell.csv", "/*_ocell.csv", "/*_slit.csv", "/*_param.json"]

    for i, xxx in enumerate(csv_title):
        for j, yyy in enumerate(glob.glob(os.getcwd() + xxx)):
            if i == 0 and j == 0:
                result["icell_path"] = yyy
            elif i == 1 and j == 0:
                result["ocell_path"] = yyy
            elif i == 2 and j == 0:
                result["slit_path"] = yyy
            elif i == 3 and j == 0:
                result["param_path"] = yyy

    return result


def read_csv(fpath: str):
    ext_type = os.path.splitext(fpath)
    if ext_type[1] == ".csv":
        result = []
        with open(fpath) as f:
            for _ in csv.reader(f):
                result.append(_)
    elif ext_type[1] == ".json":
        result = {}
        with open(fpath) as f:
            result = json.load(f)

    return result


main()

# %%
