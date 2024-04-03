name = 'cdh_185'

requires = [
    'houdini-18.5.351',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.3',
    'DXK_pipelineTools-0.23',
    'DXK_renderTools-0.26',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.1',
    'baselib-2.5',

    'dxusd_houdini-1.0.1',
    'pyalembic',
    'HOU_Base-1.0.0'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
