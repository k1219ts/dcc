import os

import DXUSD.Exporters as exp

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg

arg = exp.ATexAttrExport()
# arg.texAttrUsd = '/show/pipe/_3d/asset/asdalCityTown/branch/houseA/texture/tex/v001/tex.attr.usd'
arg.txAttrUsd = '/show/pipe/_3d/asset/bear/texture/tex/v001/tex.attr.usd'
# arg.txArgData = txArgData
exp.TexAttrExport(arg)

# def getFiles(dirpath):
#     files = list()
#     for n in os.listdir(dirpath):
#         if not n.startswith('.') and n.split('.')[-1] == 'tex':
#             if '(' in n:
#                 continue
#             files.append(n)
#     files.sort()
#     result = list()
#     vsname = None
#     for n in files:
#         if vsname:
#             if vsname != n.split('.')[0]:
#                 result.append(n)
#         else:
#             result.append(n)
#         vsname = n.split('.')[0]
#     return result
#
# dirpath = '/show/pipe/_3d/asset/bear/texture/tex/v001'
# files = getFiles(dirpath)
# for f in files:
#     print f
