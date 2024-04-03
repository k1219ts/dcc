# 3DE4.script.name: Reparametrize Lens...
# 3DE4.script.version: v1.0
# 3DE4.script.comment: Converts selected lens to a different filmback
# 3DE4.script.comment: than what the lens was calculated with.
# 3DE4.script.comment:
# 3DE4.script.comment: Currently implemented lens models:
# 3DE4.script.comment: 3DE4 Radial - Standard, Degree 4
# 3DE4.script.comment: 3DE4 Anamorphic - Standard, Degree 4
# 3DE4.script.comment: 3DE4 Anamorphic, Degree 6
# 3DE4.script.gui: Object Browser::Edit
# 3DE4.script.gui: Object Browser::Context Menu Lens

# v1.0 2018/09/24 by Unai Martinez Barredo (unaimb.com),
#                 for Jellyfish Pictures Ltd (jellyfishpictures.co.uk).
#      Math from "The Standard Models for Lens Distortion in 3DE4" whitepaper,
#      v1.3 2018/09/19 by Uwe Sassenberg, Science-D-Visions (uwe@sci-d-vis.com).

# Required 3DE Version: 3DE4 R4.

# This makes it usable as a module.
import tde4

# Equation implementations.
MODELS = {
    '3DE4 Radial - Standard, Degree 4': {
        'before': {'rho3': 'rho**3',
                   'rho4': 'rho**4'},
        'params': {'Distortion - Degree 2': 'val * rho**2',
                   'U - Degree 2': 'val * rho',
                   'V - Degree 2': 'val * rho',
                   'Quartic Distortion - Degree 4': 'val * b["rho4"]',
                   'U - Degree 4': 'val * b["rho3"]',
                   'V - Degree 4': 'val * b["rho3"]'}},
    '3DE4 Anamorphic - Standard, Degree 4': {
        'before': {'rho2': 'rho**2',
                   'rho4': 'rho**4'},
        'params': {'Cx02 - Degree 2': 'val * b["rho2"]',
                   'Cy02 - Degree 2': 'val * b["rho2"]',
                   'Cx22 - Degree 2': 'val * b["rho2"]',
                   'Cy22 - Degree 2': 'val * b["rho2"]',
                   'Cx04 - Degree 4': 'val * b["rho4"]',
                   'Cy04 - Degree 4': 'val * b["rho4"]',
                   'Cx24 - Degree 4': 'val * b["rho4"]',
                   'Cy24 - Degree 4': 'val * b["rho4"]',
                   'Cx44 - Degree 4': 'val * b["rho4"]',
                   'Cy44 - Degree 4': 'val * b["rho4"]'}},
    '3DE4 Anamorphic, Degree 6': {
        'before': {'rho2': 'rho**2',
                   'rho4': 'rho**4',
                   'rho6': 'rho**6'},
        'params': {'Cx02 - Degree 2': 'val * b["rho2"]',
                   'Cy02 - Degree 2': 'val * b["rho2"]',
                   'Cx22 - Degree 2': 'val * b["rho2"]',
                   'Cy22 - Degree 2': 'val * b["rho2"]',
                   'Cx04 - Degree 4': 'val * b["rho4"]',
                   'Cy04 - Degree 4': 'val * b["rho4"]',
                   'Cx24 - Degree 4': 'val * b["rho4"]',
                   'Cy24 - Degree 4': 'val * b["rho4"]',
                   'Cx44 - Degree 4': 'val * b["rho4"]',
                   'Cy44 - Degree 4': 'val * b["rho4"]',
                   'Cx06 - Degree 6': 'val * b["rho6"]',
                   'Cy06 - Degree 6': 'val * b["rho6"]',
                   'Cx26 - Degree 6': 'val * b["rho6"]',
                   'Cy26 - Degree 6': 'val * b["rho6"]',
                   'Cx46 - Degree 6': 'val * b["rho6"]',
                   'Cy46 - Degree 6': 'val * b["rho6"]',
                   'Cx66 - Degree 6': 'val * b["rho6"]',
                   'Cy66 - Degree 6': 'val * b["rho6"]'}}}

# Constants for the UI.
NAME = 'Reparametrize Lens...'
CUSTOM = '(User-defined)'
LABELS = ['Source Filmback Width/Height', 'Target Filmback Width/Height']
LABELMARGIN = 55
FIRSTFIELDW = 76

class ReparametrizeData(object):
    def __init__(self):
        super(ReparametrizeData, self).__init__()
        self.chosen = {}
        self.list = {}
        # TODO: Retrieve default from preferences.
        self.list[CUSTOM] = [24.576, 18.672]
        for lens in tde4.getLensList():
            self.list['"{}"'.format(
                tde4.getLensName(lens))] = (tde4.getLensFBackWidth(lens) * 10,
                                            tde4.getLensFBackHeight(lens) * 10)
        self.sorted = sorted(self.list.keys())

def reparametrize_lens(lens, model, data):
    fbw = data.chosen['fbw']
    fbh = data.chosen['fbh']
    fbw_ = data.chosen['fbw_']
    fbh_ = data.chosen['fbh_']
    rho = _fbr(fbw_, fbh_) / _fbr(fbw, fbh)
    focal = tde4.getLensFocalLength(lens)
    focus = tde4.getLensFocus(lens)
    b = {}
    for var in MODELS[model]['before']:
        b[var] = eval(MODELS[model]['before'][var])
    for param in MODELS[model]['params']:
        equ = MODELS[model]['params'][param]
        # Static distortion.
        val = tde4.getLensLDAdjustableParameter(lens, param, focal, focus)
        tde4.setLensLDAdjustableParameter(lens, param, focal, focus, eval(equ))
        # Distortion curve.
        crv = tde4.getLensLDAdjustableParameterCurve(lens, param)
        for key in tde4.getCurveKeyList(crv, False):
            pos, val = tde4.getCurveKeyPosition(crv, key)
            tde4.setCurveKeyPosition(crv, key, [pos, eval(equ)])
            # TODO: Deal with tangents.
        # Distortion 2D LUT.
        for idx in range(tde4.getLensNo2DLUTSamples(lens, param)):
            focal, focus, val = tde4.getLens2DLUTSample(lens, param, idx)
            tde4.setLens2DLUTSample(lens, param, idx, focal, focus, eval(equ))

def _fbr(fbw, fbh):
    return tde4.sqrt((fbw / 2.0)**2 + (fbh / 2.0)**2)

def filmback_widget(req, lenses, data, sanitisecbk, pickcbk, step=0):
    # Shorthands.
    suf = ('', '_')[step]
    fbo = 'fbo' + suf
    fbw = 'fbw' + suf
    fbh = 'fbh' + suf
    # Create widgets.
    tde4.addOptionMenuWidget(req, fbo, LABELS[step], '')
    tde4.addTextFieldWidget(req, fbw, '')
    tde4.addTextFieldWidget(req, fbh, '')
    # Lay out the widgets.
    tde4.setWidgetOffsets(req, fbo, LABELMARGIN, 5, 5, 5)
    tde4.setWidgetAttachModes(req, fbw, 'ATTACH_AS_IS', 'ATTACH_POSITION',
                              'ATTACH_AS_IS', 'ATTACH_AS_IS')
    tde4.setWidgetOffsets(req, fbw, LABELMARGIN, FIRSTFIELDW, 5, 5)
    tde4.setWidgetAttachModes(req, fbh, 'ATTACH_WIDGET', 'ATTACH_AS_IS',
                              'ATTACH_OPPOSITE_WIDGET', 'ATTACH_AS_IS')
    tde4.setWidgetLinks(req, fbh, fbw, '', fbw, '')
    tde4.setWidgetOffsets(req, fbh, 5, 5, 0, 5)
    # Set widget callbacks and initialise them.
    for name in (fbw, fbh):
        tde4.setWidgetCallbackFunction(req, name, sanitisecbk)
    tde4.modifyOptionMenuWidget(req, fbo, LABELS[step], *data.sorted)
    tde4.setWidgetCallbackFunction(req, fbo, pickcbk)
    idx = 1
    if step == 1:
        if lenses:
            idx = data.sorted.index(sorted('"{}"'.format(
                tde4.getLensName(x)) for x in lenses)[0]) + 1
    else:
        idx = len(data.list)
    tde4.setWidgetValue(req, fbo, str(idx))
    list_pick(req, fbo, data)

def list_pick(req, name, data):
    outs = ['fbw', 'fbh']
    if name[-1] == '_':
        outs = [x + '_' for x in outs]
    lens = menulens(req, name, data)
    vals = data.list[lens]
    edit = lens == CUSTOM
    for idx, out in enumerate(outs):
        tde4.setWidgetValue(req, out, str(vals[idx]))
        sanitise_float_field(req, out, data, edit=edit)
        tde4.setWidgetSensitiveFlag(req, out, edit)

def menulens(req, name, data):
    return data.sorted[tde4.getWidgetValue(req, name) - 1]

def sanitise_float_field(req, name, data, edit=True):
    try:
        val = float(tde4.getWidgetValue(req, name).split(' ', 1)[0])
    except ValueError:
        val = 0.0
    data.chosen[name] = val
    if edit:
        data.list[CUSTOM]['h' in name] = val
    # TODO: Take unit conversion and defaults into account.
    tde4.setWidgetValue(req, name, '{:.4f} mm'.format(val))

if __name__ == '__main__':
    # Get lenses to act on, accounting for context menu first,
    # or getting the selected ones if it was chosen from the Edit menu instead.
    lens = tde4.getContextMenuObject()
    if lens:
        inlenses = [lens]
    else:
        inlenses = tde4.getLensList(True)

    # Get distortion model for each lens, reject if not implemented.
    lenses = {}
    reject = []
    for lens in inlenses:
        model = tde4.getLensLDModel(lens)
        if not model in MODELS:
            reject.append(lens)
            continue
        lenses[lens] = model

    # Build GUI.
    ret = None
    if lenses:
        req = tde4.createCustomRequester()
        data = ReparametrizeData()
        _sanitise = lambda x, y, z: sanitise_float_field(x, y, data)
        _listpick = lambda x, y, z: list_pick(x, y, data)
        filmback_widget(req, lenses, data, '_sanitise', '_listpick')
        filmback_widget(req, lenses, data, '_sanitise', '_listpick', step=1)
        ret = tde4.postCustomRequester(req, NAME, 430, 170, 'Ok', 'Cancel')

    # Show message if no lens was selected.
    elif not reject:
        tde4.postQuestionRequester(NAME, 'No lens selected!', 'Ok')

    # Run.
    if ret == 1:
        for lens in lenses:
            reparametrize_lens(lens, lenses[lens], data)
        tde4.deleteCustomRequester(req)

    # Show information for rejected lenses.
    if (not lenses or ret == 1) and reject:
        plu = len(reject) > 1
        reject = ['"{}"'.format(tde4.getLensName(x)) for x in reject]
        lst = reject.pop()
        if plu:
            lst = ' and '.join((', '.join(reject), lst))
        msg = ("Couldn't reparametrize {} because {} distortion model{} "
               "ha{}n't been implemented in this tool.").format(
                   lst, ('its', 'their')[plu], ('', 's')[plu],
                   ('s', 've')[plu])
        tde4.postQuestionRequester(NAME, msg, 'Ok')
