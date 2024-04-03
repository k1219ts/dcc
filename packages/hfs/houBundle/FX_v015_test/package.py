name = 'FX_v015'

requires = [
    'houdini-18.5.596',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.3',
    'DXK_pipelineTools-0.23',
    'DXK_renderTools-0.26',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.11',
    'DXK_waterTools-0.1',
    'baselib-2.5',
    'pyalembic',
    'dxusd_houdini-1.0.5',
    'HOU_Base-1.0.0',
    'HOU_Feather-1.0.1',
    'HOU_Ocean-1.0.0',
    'DX_Environment-1.0.0',
    #'rfh-23.5'
    #'rfh-23.4',
    #'AxisTool-0.0.7'
]

def commands():
    env.BUNDLE_NAME.set(this.version)