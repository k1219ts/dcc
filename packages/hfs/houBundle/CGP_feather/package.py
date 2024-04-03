name = 'FX_v013_USD'

requires = [
    'houdini-17.5.229',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.2',
    'DXK_pipelineTools-0.19',
    'DXK_renderTools-0.24',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.1',
    'DXK_usdTools',
    'usd_houdini',
    'baselib-2.5',
    'Feather_houdini'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
