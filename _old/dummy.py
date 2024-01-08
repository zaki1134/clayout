class ViewHelper:
    def SetSectionPlane():
        pass

    def SetSketchPlane():
        pass

    def SetViewMode():
        pass

    def SetObjectVisibility():
        pass


class DocumentHelper:
    def GetAllDocObjects():
        pass


class GetRootPart:
    def Bodies():
        pass

    def DatumPlanes():
        pass

    def GetAllBodies():
        pass


class NamedSelection:
    def Create():
        pass

    def Rename():
        pass

    def GetGroups():
        pass


class Selection:
    def Create():
        pass

    def Empty():
        pass


# -----------------------------------
class SketchLine:
    def Create():
        pass


class SketchArc:
    def CreateSweepArc():
        pass


class List:
    def dummy():
        pass


class IDesignBody:
    def dummy():
        pass


class Delete:
    def Execute():
        pass


class InteractionMode:
    def Sketch():
        pass

    def Solid():
        pass


class Point:
    def Create():
        pass


class Point2D:
    def Create():
        pass


class MM:
    def dummy():
        pass


class SketchCircle:
    def Create():
        pass


class DetachFaces:
    def Execute():
        pass


class VisibilityType:
    def Hide():
        pass

    def Show():
        pass


# -----------------------------------
class SetColorOptions:
    def FaceColorTarget():
        pass


class FaceColorTarget:
    def Body():
        pass


class ColorHelper:
    def SetColor():
        pass


class Color:
    def FromArgb():
        pass


# -----------------------------------
class SketchRectangle:
    def Create():
        pass


# -----------------------------------
class ExtrudeFaceOptions:
    def ExtrudeType():
        pass


class ExtrudeType:
    def ForceIndependent():
        pass

    def Forcecut():
        pass


class ExtrudeFaces:
    def Execute():
        pass


# -----------------------------------
class FixInterference:
    def FindAndFix():
        pass


# -----------------------------------
class MirrorOptions:
    def CreateRelationships():
        pass


class Mirror:
    def Execute():
        pass


# -----------------------------------
class DatumPlaneCreator:
    def Create():
        pass


class Frame:
    def Create():
        pass


class Direction:
    def Create():
        pass


class Plane:
    def PlaneXY():
        pass

    def Create():
        pass
