name = 'CROWD18.5'

requires = [
    'pyalembic',
    'houdini-18.5.351',
    # 'rfh-23.5',
    # 'DX_Environment-1.0.0',
    'dxusd_houdini-1.0.5',
    # 'HOU_Base-1.0.0',
    # 'HOU_Feather-1.0.1'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
