name = 'pubtest'

requires = [
    'houdini-18.5.351',
    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.4',
    'DXK_pipelineTools-0.23',
    'DXK_renderTools-0.26',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.11',
    'DXK_waterTools-0.1',
    'baselib-2.5',
    'pyalembic',
    'dxusd_houdini-1.0.6',
    'HOU_Base-1.0.0',
    'HOU_Feather-1.0.2',
    'HOU_Ocean-1.0.0',
    'DX_Environment-1.0.0',
#    'ffmpeg'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
