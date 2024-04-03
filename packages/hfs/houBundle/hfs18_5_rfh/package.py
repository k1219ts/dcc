name = 'hfs18_5_rfh'

requires = [
    'houdini-18.5.351',
    # 'DX_pipelineTools-2.2',
    # 'DXK_renderfarm-3.1',
    # 'DXK_pipelineTools-0.19',
    # 'DXK_renderTools-0.23',
    # 'DXK_old_Otls-0.1',
    # 'DXK_sceneSetupMng-0.2',
    # 'DXK_info-0.1',
    # 'usd_houdini',
    'rfh-23.5',
    'DX_Environment-1.0.0',
    'dxusd_houdini-1.0.1',
    'pyalembic'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
