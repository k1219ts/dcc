import shutil, os

import DXUSD.Exporters as exp

import DXUSD.Vars as var
import DXUSD.Utils as utl
import DXUSD.Message as msg


class DCC_ModelExport(exp.ModelExport):
    def Exporting(self):
        print(self.arg.aaaaaa)
        # --------------------------------------------------------------------------
        # Prepare for test
        outPath = '/show/pipe/_3d/asset/lion/model/v001'
        orgPath = '%s/test'%outPath
        orgs = [
            '%s/lion_model_GRP.high_geom.usd'
        ]

        for org in orgs:
            if os.path.exists(org%outPath):
                os.remove(org%outPath)
            shutil.copy2(org%orgPath, org%outPath)

        paths = ['/show/pipe/_3d/asset/lion/texture/tex/v001/tex.attr.usd',
                 '/show/pipe/_3d/asset/lion/texture/tex/tex.usd']
        # for path in paths:
        #     if os.path.exists(path):
        #         os.remove(path)

        return var.SUCCESS


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # Set arguments
    arg = exp.AModelExport(show='pipe')
    # arg.asset = 'animal'
    # arg.branch = 'lion'
    arg.ver   = 'v001'
    arg.srclyr.name = 'lion_model_GRP'
    # arg.asset = 'lion'
    # arg.srclyr[0] = True
    # arg.srclyr.mid = False
    # arg.srclyr[var.T.LOW] = True

    # --------------------------------------------------------------------------
    # Run Exporter
    DCC_ModelExport(arg)
