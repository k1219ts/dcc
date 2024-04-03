name = 'FX_v014_RMAN'

requires = [
    'houdini-18.0.499',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.2',
    'DXK_pipelineTools-0.21',
    'DXK_renderTools-0.24',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.11',
    'DXK_waterTools-0.1',
    'axiom-0.1',
    'baselib-2.5',
    'rfh-23.4'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
