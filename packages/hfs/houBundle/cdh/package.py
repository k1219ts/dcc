name = 'cdh'

requires = [
    'houdini-17.5.229',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.3',
    'DXK_pipelineTools-0.22',
    'DXK_renderTools-0.26',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.1',
    'DXK_usdTools-0.1',
    'usd_houdini',
    'baselib-2.5'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
