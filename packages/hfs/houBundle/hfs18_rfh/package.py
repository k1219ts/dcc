name = 'hfs18_rfh'

requires = [
    'houdini-18.0.499',
    # 'DX_pipelineTools-2.2',
    # 'DXK_renderfarm-3.1',
    # 'DXK_pipelineTools-0.19',
    # 'DXK_renderTools-0.23',
    # 'DXK_old_Otls-0.1',
    # 'DXK_sceneSetupMng-0.2',
    # 'DXK_info-0.1',
    # 'usd_houdini',
    # 'baselib-1.0',
    'baselib-2.5',
    'rfh-23.2',
    'DX_Environment-1.0.0',
    'dxusd_houdini-1.0.1',
    'pyalembic'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
