'''
Basefount Miarmy support scripts

--
[ Export alembic points for McdAgents ]
AgentPointsExport(filename, start, end)

--
[ Export alembic points for selected nodes ]
exp = PointsExport(filename, nodes, name, start, end)
exp.doIt()

--
[ Export frame ribs ]
exp = ArmyRender(1, 1001, 1030)
exp.doIt()

[ Export agent prim asset ribs ]
exp = ArmyRender(2, 0, 0)
exp.doIt()

python
    ExportRib(1, 1001, 1030)
    ExportRib(2, 0, 0)
mel
    DxArmyExportRib 1 1001 1030
    DxArmyExportRib 2 0 0

--
[ Export Rib Tractor Spool ]
options = {
    'm_chunk': dispatch frame range size
    'm_mayafile': maya filename
    'm_outdir': output path
    'm_start': start frame
    'm_end': end frame
}
ExportRibSpool(options)

--
Delete modules
src = list()
for i in sys.modules:
    if i.find('dxArmy') > -1:
        src.append(i)
for i in src:
    del sys.modules[i]
'''

from PointsExport import *
from RibExport import *
from JobScript import *
from RManAttrCtrl import *

__all__ = (
    'PointsExport', 'AgentPointsExport',
    'ExportRib',
    'ExportRibSpool', 'SampleSpool',
    'AgentTextureAttributes'
)
