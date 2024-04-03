from fnpxr import Usd, UsdGeom, Sdf


def MakeInitialLayer(filename, usdformat='usda', clear=False, comment=None):
    outLayer = Sdf.Layer.FindOrOpen(filename)
    if not outLayer:
        outLayer = Sdf.Layer.CreateNew(filename, args={'format': usdformat})
    if clear:
        outLayer.Clear()

    if comment:
        curcomment = outLayer.comment
        if curcomment:
            if not comment in curcomment:
                comment = curcomment + ', ' + comment
        outLayer.comment = comment

    return outLayer


def MakeInitialStage(filename, usdformat='usda', clear=False, comment=None, fr=(None, None), fps=24.0):
    outLayer = MakeInitialLayer(filename, usdformat, clear, comment)
    stage = Usd.Stage.Open(outLayer)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    TimeCodeSetup(stage, fr, fps)
    return stage


def TimeCodeSetup(stage, fr=(None, None), fps=24.0):
    '''
    start is min value, end is max value
    '''
    if not (fr[0] != None and fr[1] != None):
        return

    if fr[0] != fr[1]:
        stage.SetStartTimeCode(fr[0])
        stage.SetEndTimeCode(fr[1])

        stage.SetFramesPerSecond(fps)
        stage.SetTimeCodesPerSecond(fps)
