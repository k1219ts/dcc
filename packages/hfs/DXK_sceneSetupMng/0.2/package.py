name = 'DXK_sceneSetupMng'

def commands():
    env.HOUDINI_TOOLBAR_PATH.append('{this.root}/toolbar')
    env.PYTHONPATH.append('{this.root}/scripts')
    env.PYTHONPATH.append('/netapp/backstage/pub/apps/inventory/src/dev')
