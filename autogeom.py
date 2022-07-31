import csv
import os
import glob
import json
from dummy import *


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

    # initialize
    _ = Selection.Create(DocumentHelper.GetAllDocObjects())
    if _.Count != 0:
        Delete.Execute(_)

    # make icell
    harf_length = 0.5 * param_set["length_prod"]
    n_icell = SolidGroup(title="n_icell")
    n_icell.CicleGroup(pos_i, param_set["diameter_icell"])
    n_icell.Extrude(harf_length)
    n_icell.NamedSelection()
    n_icell.Hide()

    # make membrane
    if not temp_isclose(param_set["thickness_bot"], 0.0):
        d_out = param_set["diameter_icell"]
        d_in = d_out - 2.0 * param_set["thickness_bot"]
        n_bot = SolidGroup(title="n_bot")
        n_bot.TorusGroup(pos_i, d_in, d_out)
        n_bot.Extrude(harf_length)
        n_bot.NamedSelection()
        n_bot.Hide()

    if not temp_isclose(param_set["thickness_mid"], 0.0):
        d_out = param_set["diameter_icell"] - 2.0 * param_set["thickness_bot"]
        d_in = d_out - 2.0 * param_set["thickness_mid"]
        n_mid = SolidGroup(title="n_mid")
        n_mid.TorusGroup(pos_i, d_in, d_out)
        n_mid.Extrude(harf_length)
        n_mid.NamedSelection()
        n_mid.Hide()

    if not temp_isclose(param_set["thickness_top"], 0.0):
        d_out = param_set["diameter_icell"] - 2.0 * (param_set["thickness_bot"] + param_set["thickness_mid"])
        d_in = d_out - 2.0 * param_set["thickness_top"]
        n_top = SolidGroup(title="n_top")
        n_top.TorusGroup(pos_i, d_in, d_out)
        n_top.Extrude(harf_length)
        n_top.NamedSelection()
        n_top.Hide()

    # make ocell
    offset = -(param_set["length_seal"] + param_set["length_slit"])
    n_ocell = SolidGroup(pos_plane=offset, title="n_ocell")
    n_ocell.OctGroup(pos_o, param_set)
    n_ocell.Extrude(harf_length + offset)
    n_ocell.NamedSelection()
    n_ocell.Hide()

    # make slit
    offset = -param_set["length_seal"]
    n_slit = SolidGroup(pos_plane=offset, title="n_slit")
    n_slit.RectangleGroup(pos_s, param_set)
    n_slit.Extrude(param_set["length_slit"])
    cutobj = SolidGroup()
    cutobj.Torus(param_set["diameter_prod"], 2.0 * param_set["diameter_prod"])
    cutobj.Extrude(param_set["length_prod"], True)
    n_slit.NamedSelection()
    n_slit.Hide()

    # make base
    n_base = SolidGroup(title="n_base")
    n_base.Circle(param_set["diameter_prod"])
    n_base.Extrude(harf_length)
    n_base.NamedSelection()

    # fix interference
    n_icell.Show()
    FixInterference.FindAndFix()
    if not temp_isclose(param_set["thickness_top"], 0.0):
        n_top.Show()
        FixInterference.FindAndFix()
    if not temp_isclose(param_set["thickness_mid"], 0.0):
        n_mid.Show()
        FixInterference.FindAndFix()
    if not temp_isclose(param_set["thickness_bot"], 0.0):
        n_bot.Show()
        FixInterference.FindAndFix()
    n_ocell.Show()
    FixInterference.FindAndFix()
    n_slit.Show()
    FixInterference.FindAndFix()

    # set color
    set_color()

    # cut quarter
    cutobj = SolidGroup(1.0)
    cutobj.Quarter(param_set["diameter_prod"])
    cutobj.Extrude(param_set["length_prod"], True)

    # copy mirror
    org = Point.Create(MM(0.0), MM(0.0), MM(-0.5 * param_set["length_prod"]))
    DatumPlaneCreator.Create(org, Direction.DirZ, False, None)

    xxx = GetRootPart().DatumPlanes[0]
    mirror_plane = Selection.Create(xxx)
    opt = MirrorOptions()
    opt.CreateRelationships = False
    for bodies in GetRootPart().GetAllBodies():
        _ = Selection.Create(bodies)
        Mirror.Execute(_, mirror_plane, opt, None)
    xxx.Delete()

    del pos_i, pos_o, pos_s, param_set


class SolidGroup:
    def __init__(self, pos_plane=0.0, title="abc"):
        self.pos_plane = pos_plane
        self.title = title

    def sketchmode(func):
        def wrapper(self, *args, **kwargs):
            # pre
            org = Point.Create(0.0, 0.0, MM(self.pos_plane))
            frm = Frame.Create(org, Direction.Create(1, 0, 0), Direction.Create(0, 1, 0))
            ViewHelper.SetSketchPlane(Plane.Create(frm))

            func(self, *args, **kwargs)

            # post
            ViewHelper.SetViewMode(InteractionMode.Solid, None)

        return wrapper

    @sketchmode
    def Circle(self, diameter=1.0):
        org = Point2D.Create(MM(0.0), MM(0.0))
        SketchCircle.Create(org, MM(0.5 * diameter))

    @sketchmode
    def CicleGroup(self, pos_i, diameter=1.0):
        for x, y in pos_i:
            center = Point2D.Create(MM(x), MM(y))
            SketchCircle.Create(center, MM(0.5 * diameter))

    @sketchmode
    def OctGroup(self, pos_o, param_set):
        def LineOct(c_pitch, o_rib, o_hight, x1, y1, center=(0.0, 0.0)):
            point = []
            dx = c_pitch - o_rib - 2.0 * x1
            dy = o_hight - 2.0 * y1
            width = c_pitch - o_rib

            # oct point
            point.append([center[0] + 0.5 * (c_pitch - o_rib), center[1] + 0.5 * o_hight - y1])
            point.append([point[0][0] - x1, point[0][1] + y1])

            point.append([point[1][0] - dx, point[1][1]])
            point.append([point[0][0] - width, point[0][1]])

            point.append([point[0][0] - width, point[0][1] - dy])
            point.append([point[1][0] - dx, point[1][1] - o_hight])

            point.append([point[1][0], point[1][1] - o_hight])
            point.append([point[0][0], point[0][1] - dy])

            point.append([point[0][0], point[0][1]])

            for i in range(8):
                aaa = Point2D.Create(MM(point[i][0]), MM(point[i][1]))
                bbb = Point2D.Create(MM(point[i + 1][0]), MM(point[i + 1][1]))
                SketchLine.Create(aaa, bbb)

        for x, y in pos_o:
            LineOct(
                param_set["diameter_icell"] + param_set["thickness_rib"],
                param_set["thickness_rib_ocell"],
                param_set["hight_ocell"],
                param_set["width_x1_ocell"],
                param_set["hight_y1_ocell"],
                center=(x, y),
            )

    @sketchmode
    def Torus(self, d_in=1.0, d_out=2.0):
        center = Point2D.Create(MM(0.0), MM(0.0))
        SketchCircle.Create(center, MM(0.5 * d_in))
        SketchCircle.Create(center, MM(0.5 * d_out))

    @sketchmode
    def TorusGroup(self, pos_i, d_in=1.0, d_out=2.0):
        for x, y in pos_i:
            center = Point2D.Create(MM(x), MM(y))
            SketchCircle.Create(center, MM(0.5 * d_in))
            SketchCircle.Create(center, MM(0.5 * d_out))

    @sketchmode
    def RectangleGroup(self, pos_s, param_set):
        def LineRect(yyy, t_slit, width):
            p1 = Point2D.Create(MM(-width), MM(yyy + 0.5 * t_slit))
            p2 = Point2D.Create(MM(width), MM(yyy + 0.5 * t_slit))
            p3 = Point2D.Create(MM(width), MM(yyy - 0.5 * t_slit))
            SketchRectangle.Create(p1, p2, p3)

        for y in pos_s:
            LineRect(y, param_set["thickness_slit"], 0.6 * param_set["diameter_prod"])

    @sketchmode
    def Quarter(self, diameter=1.0):
        org = Point2D.Create(MM(0.0), MM(0.0))
        aaa = Point2D.Create(MM(0.0), MM(3.0 * diameter))
        SketchLine.Create(org, aaa)

        bbb = Point2D.Create(MM(3.0 * diameter), MM(0.0))
        SketchLine.Create(org, bbb)

        SketchArc.CreateSweepArc(org, aaa, bbb, False)

    def Extrude(self, length=1.0, cut=False):
        for bodies in GetRootPart().GetAllBodies():
            if ViewHelper.GetObjectVisibility(bodies):
                facebodies = Selection.Create(bodies)

        opt = ExtrudeFaceOptions()
        if not cut:
            opt.ExtrudeType = ExtrudeType.ForceIndependent
        else:
            opt.ExtrudeType = ExtrudeType.ForceCut
        ExtrudeFaces.Execute(facebodies.ConvertToFaces(), MM(-length), opt)

    def NamedSelection(self):
        _ = List[IDesignBody]()
        for bodies in GetRootPart().GetAllBodies():
            if ViewHelper.GetObjectVisibility(bodies):
                _.Add(bodies)

        NamedSelection.Create(Selection.Create(_), Selection.Empty())
        NamedSelection.Rename("Group1", self.title)

    def Show(self):
        for ns in NamedSelection.GetGroups():
            if ns.GetName() == self.title:
                _ = Selection.Create(ns.Members)
                ViewHelper.SetObjectVisibility(_, VisibilityType.Show, False, False)

    def Hide(self):
        for ns in NamedSelection.GetGroups():
            if ns.GetName() == self.title:
                _ = Selection.Create(ns.Members)
                ViewHelper.SetObjectVisibility(_, VisibilityType.Hide, False, False)


def set_color():
    opt = SetColorOptions()
    opt.FaceColorTarget = FaceColorTarget.Body
    for ns in NamedSelection.GetGroups():
        _ = Selection.Create(ns.Members)
        if ns.GetName() == "n_icell":
            ColorHelper.SetColor(_, opt, Color.FromArgb(128, 128, 255, 255))
        elif ns.GetName() == "n_ocell":
            ColorHelper.SetColor(_, opt, Color.FromArgb(128, 0, 255, 128))
        elif ns.GetName() == "n_bot":
            ColorHelper.SetColor(_, opt, Color.FromArgb(255, 128, 128, 128))
        elif ns.GetName() == "n_mid":
            ColorHelper.SetColor(_, opt, Color.FromArgb(255, 128, 128, 192))
        elif ns.GetName() == "n_top":
            ColorHelper.SetColor(_, opt, Color.FromArgb(255, 255, 128, 64))
        elif ns.GetName() == "n_slit":
            ColorHelper.SetColor(_, opt, Color.FromArgb(128, 0, 128, 255))
        elif ns.GetName() == "n_base":
            ColorHelper.SetColor(_, opt, Color.FromArgb(255, 192, 192, 192))


def search_path():
    result = {}
    csv_title = ["/*_icell.csv", "/*_ocell.csv", "/*_slit.csv", "/*_param.json"]

    for i, title in enumerate(csv_title):
        for j, fullpath in enumerate(glob.glob(os.getcwd() + title)):
            if i == 0 and j == 0:
                result["icell_path"] = fullpath
            elif i == 1 and j == 0:
                result["ocell_path"] = fullpath
            elif i == 2 and j == 0:
                result["slit_path"] = fullpath
            elif i == 3 and j == 0:
                result["param_path"] = fullpath

    return result


def read_csv(filepath):
    # slit lengthが実装されたらifいらない
    ext_type = os.path.splitext(filepath)
    if ext_type[1] == ".csv":
        result = []
        with open(filepath) as f:
            for row in csv.reader(f):
                if len(row) != 1:
                    _ = [float(row[0]), float(row[1])]
                else:
                    _ = float(row[0])
                result.append(_)

    elif ext_type[1] == ".json":
        result = {}
        with open(filepath) as f:
            result = json.load(f)

    return result


def temp_isclose(aaa, bbb):
    """
    Alternative to math.isclose.
    """
    return abs(aaa - bbb) <= max(1e-9 * max(abs(aaa), abs(bbb)), 0.0)


main()
print("***** Done ******")
