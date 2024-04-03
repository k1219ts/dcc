name = 'cgsup'

requires = [
    # 'houdini-16.5.405',
    'houdini-17.5.460',
    'usd_houdini',
    #'rfh-22.5',
    #'DX_pipelineTools-2.2',
    #'DXK_old_Otls-0.1'
]

def commands():
    env.BUNDLE_NAME.set(this.version)
