#
#
# 3DE4.script.name:  Import SphereGrid Model...
#
# 3DE4.script.version:  v1.0
#
# 3DE4.script.gui:  Main Window::Dexter
#
# 3DE4.script.comment:  Import spheregrid model.
#
#
SPHEREGRID = '/dexter/Cache_DATA/MMV/asset/spheregrid.obj'

pg = tde4.getCurrentPGroup()
m = tde4.create3DModel(pg, 0)
tde4.importOBJ3DModel(pg, m, SPHEREGRID)
tde4.set3DModelName(pg, m, 'spheregrid.obj')
tde4.set3DModelReferenceFlag(pg, m, 1)
