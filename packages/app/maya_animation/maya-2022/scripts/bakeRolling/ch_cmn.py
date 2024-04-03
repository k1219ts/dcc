import pymel.core as pm
import pymel.core.nodetypes as nt
import pymel.core.datatypes as dt


def confirmDialog(msg, btype=0, error=True):
    btypes = [
        ['Ok'],
        ['Yes', 'No']
    ]
    msgTypoe = 'Error' if error else 'Confirm'
    dlg = pm.confirmDialog(
            title=msgTypoe,
            message=msg, button=btypes[btype]
            )
    if error and dlg in [v[-1] for v in btypes]:
        raise BaseException

    return dlg

def shortNameOf(obj, ns=False):
    name = obj.name() if isinstance(obj, pm.PyNode) else str(obj)
    name = name.split('|')[-1]

    if ns:
        return name

    return name.split(':')[-1]
