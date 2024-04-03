from maya import OpenMayaRender
from maya import OpenMayaUI

renderer = OpenMayaRender.MHardwareRenderer
glFT = renderer.theRenderer().glFunctionTable()

class Shape(object):
    def coords(self):
        return []
    def draw(self, m3dview):
        m3dview.beginGL()
        for segmentcoords in self.coords():
            glFT.glBegin(OpenMayaRender.MGL_LINE_STRIP)
            for coords in segmentcoords:
                glFT.glVertex3f(*coords)
            glFT.glEnd()
        m3dview.endGL()

class Cross(Shape):
    def coords(self):
        return [
            ((-10, 0, 0), (10, 0, 0)),
            ((0, -10, 0), (0, 10, 0)),
            ((0, 0, -10), (0, 0, 10)) ]

class Square(Shape):
    def coords(self):
        return [
            ((20, 20, 0),
             (20, -20, 0),
             (-20, -20, 0),
             (-20, 20, 0),
             (20, 20, 0)) ]

m3dview = OpenMayaUI.M3dView.active3dView()
m3dview.beginOverlayDrawing()
Square().draw(m3dview)
Cross().draw(m3dview)
m3dview.endOverlayDrawing()
