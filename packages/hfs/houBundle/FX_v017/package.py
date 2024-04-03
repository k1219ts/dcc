name = 'FX_v017'

requires = [
    'houdini-18.5.351',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-4.0',
    'DXK_pipelineTools-0.24',
    'DXK_renderTools-0.27',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.11',
    #'DXK_waterTools-0.1',
    'baselib-2.5',
    'pyalembic',
    'dxusd_houdini-1.0.6',
    'HOU_Base-1.1.0',
    'HOU_Feather-1.0.2',
    'HOU_Ocean-1.0.0',
    'DX_Environment-1.0.0',
    'ffmpeg',
    'ocio_configs-1.2'
    #'rfh-23.5'
    #'rfh-23.4',
    #'AxisTool-0.0.7'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
