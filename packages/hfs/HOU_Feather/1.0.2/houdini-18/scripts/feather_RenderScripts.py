#coding:utf-8
from __future__ import print_function

import hou
import HOU_Base.NodeUtils as ntl


def PreRenderScript(this):
    # find feahter designers
    # exporttype = this.parm('feather_exporttype').evalAsInt()
    # designers  = []
    #
    # if exporttype == 0: # designer
    #     node = this.parm('feather_designerpath').evalAsNode()
    #     if node.type().name() == 'merge':
    #         ntl.RetrieveByNodeType(node, 'dxSOP_FeatherDesigner',
    #                                designers, firstMatch=False)
    #     elif node.type().name() == 'dxSOP_FeatherDesigner':
    #         designers.append(node.path())
    # elif exporttype == 1: # groomer
    #     node = this.parm('feather_groomerpath').evalAsNode()
    #     node = node.inputs()[0]
    #     ntl.RetrieveByNodeType(node, 'dxSOP_FeatherDesigner',
    #                            designers, firstMatch=False)
    # 
    # this.setUserData('feather_designers', ' '.join(designers))
    pass

def PostRenderScript(this):
    pass
