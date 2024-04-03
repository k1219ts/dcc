name = 'speedtree'
version = '7.1.1'

def commands():
    INSTALL_PATH='/opt/SpeedTree/7.1.1'

    env.RLM_LICENSE.set('5053@10.0.0.170')

    env.LD_LIBRARY_PATH.append('{}/linux/lib'.format(INSTALL_PATH))
    env.PATH.append('{}/linux'.format(INSTALL_PATH))

    unsetenv('QT_PLUGIN_PATH')

    alias('speedtree', 'SpeedTree\ Modeler\ Cinema\ Eval')
