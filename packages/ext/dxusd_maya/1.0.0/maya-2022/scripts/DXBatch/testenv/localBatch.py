
from DXBatch import BatchMain

def aniBatchExport():
    sceneFile = '/show/pipe/template/fox/CLF_0050_ani_v001_sample.mb'
    rigNode = 'fox:fox_rig_GRP'

    # parser.add_argument('-f', '--file', type=str, help='Maya filename.')
    # parser.add_argument('-o', '--outDir', type=str, help='Cache out directory.')
    # parser.add_argument('-u', '--user', type=str, help='User name')
    #
    # # TimeRange argument
    # parser.add_argument('-fr', '--frameRange', type=int, nargs=2, default=(0, 0), help='frame range, (start, end)')
    # parser.add_argument('-fs', '--frameSample', type=float, default=1.0, help='frame step size default = 1.0')
    #
    # # Acting argument
    # parser.add_argument('-p', '--process', type=str, choices=['geom', 'comp'],
    #                     help='task export when choice process, [geom, comp]')
    # parser.add_argument('-h', '--host', type=str, choices=['local', 'tractor'],
    #                     help='if host local, cache export. other option is "tractor" spool')
    #
    # # task argument
    # #   Rig Out
    # parser.add_argument('-m', '--mesh', type=str, nargs='*',
    #                     help='export namspace:nodeName of dxRigNode \nex) --mesh v005=nsLayer1:node_rig_GRP v004=nsLayer2:node_rig_GRP')
    # parser.add_argument('-ru', '--rigUpdate', type=bool, action='store_true', default=False,
    #                     help='if True, using rig latest version.')
    # #   Sim Out
    # parser.add_argument('-sm', '--simMesh', type=str, nargs='*',
    #                     help='export namspace:nodeName of dxRigNode \nex) --simMesh v005=nsLayer1:node_rig_GRP v004=nsLayer2:node_rig_GRP')
    # #   Layout Out
    # parser.add_argument('-l', '--layout', type=str, nargs='*',
    #                     help='export nodeName of dxBlock \nex) --layout v005=node_set v004=node1_set')
    # #   Camera Out
    # parser.add_argument('-c', '--camera', type=str, nargs='*',
    #                     help='export nodeName of dxCamera \nex) --camera v005=dxCamera1 v004=dxCamera2')
    # #   Crowd Out
    # parser.add_argument('-cw', '--crowd', type=str, nargs='*',
    #                     help='export nodeName of dxCamera \nex) --crowd v005=dxCamera1 v004=dxCamera2')
    # #   Groom Out
    # parser.add_argument('-g', '--groom', action='count', default=0, help='if groom, export groom after exported mesh')
    # parser.add_argument('-og', '--onlyGroom', action='count', default=0,
    #                     help='if groom, export groom after exported mesh')