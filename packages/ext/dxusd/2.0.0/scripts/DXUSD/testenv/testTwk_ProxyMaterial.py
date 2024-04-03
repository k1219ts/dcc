import DXUSD.Tweakers as twk

#-------------------------------------------------------------------------------
# 1
arg = twk.AProxyMaterial()
arg.mtlDir = '/show/pipe/_3d/asset/bear/texture/proxy/v001'
if arg.Treat():
    print arg
    TPM = twk.ProxyMaterial(arg)
    TPM.DoIt()
