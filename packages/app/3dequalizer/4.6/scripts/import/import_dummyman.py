#
#
# 3DE4.script.name:  Import Dummy Model...
#
# 3DE4.script.version:  v1.0.1
#
# 3DE4.script.gui:  Main Window::Dexter::Import Data
#
# 3DE4.script.comment:  Import Dummy model.
#
# 1.0.1 correct baseMan
SPHEREGRID = '/stdrepo/MMV/asset/model/baseman/baseMan_170cm.obj'

pg = tde4.getCurrentPGroup()
m = tde4.create3DModel(pg, 0)
tde4.importOBJ3DModel(pg, m, SPHEREGRID)
tde4.set3DModelName(pg, m, 'baseMan_170cm.obj')
tde4.set3DModelReferenceFlag(pg, m, 1)
