name = 'pubtest'

requires = [
    # 'houdini-18.0.499',
    # 'baselib-2.5',
    # 'rfh-23.2',

    'houdini-18.5.351',
    'baselib-2.5',
    'rfh-23.5',

    'dxusd_houdini-1.0.5',
    'pyalembic',
    'HOU_Base-1.0.0',
    'HOU_Feather-1.0.1',
    'HOU_Ocean-1.0.0',
    'DX_Environment-1.0.0',

    'DX_pipelineTools-2.2',
    'DXK_renderfarm-3.3',
    'DXK_pipelineTools-0.23',
    'DXK_renderTools-0.26',
    'DXK_old_Otls-0.1',
    'DXK_sceneSetupMng-0.2',
    'DXK_info-0.11',
    'DXK_waterTools-0.1']

def commands():
    env.BUNDLE_NAME.set(this.version)
