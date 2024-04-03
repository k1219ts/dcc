import sys
import os

from pxr import Usd, UsdGeom, Sdf

def _forceDefaultPrim(node):
    templateFile=node.evalParm('outtemplatefile1')
    print("Target file ->" + templateFile)
    stage = Usd.Stage.Open(templateFile)
    getPrim = UsdGeom.Xform.Define(stage,Sdf.Path("/Geom")).GetPrim()
    stage.SetDefaultPrim(getPrim)
    stage.Save()

def helloworld():
    print("Hello Muad'dib")