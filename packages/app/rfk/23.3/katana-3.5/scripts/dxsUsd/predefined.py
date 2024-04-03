from Katana import RenderingAPI
from fnpxr import Sdf

_KATANA_RENDER_MAP_ = {
    'diskRender': 'kRenderMethodTypeDiskRender',
    'liveRender': 'kRenderMethodTypeLiveRender',
    'previewRender': 'kRenderMethodTypePreviewRender',
    1: 'kRenderPluginApiVersion',
    2: 'kRendererInfoApiVersion',
    'driver': 'kRendererObjectTypeDriver',
    'filter': 'kRendererObjectTypeFilter',
    'outputChannel': 'kRendererObjectTypeOutputChannel',
    'outputChannelAttrHints': 'kRendererObjectTypeOutputChannelAttrHints',
    'outputChannelCustomParam': 'kRendererObjectTypeOutputChannelCustomParam',
    'renderOutput': 'kRendererObjectTypeRenderOutput',
    'shader': 'kRendererObjectTypeShader',
    3: 'kRendererObjectValueTypeBoolean',
    0: 'kRendererObjectValueTypeByte',
    5: 'kRendererObjectValueTypeColor3',
    6: 'kRendererObjectValueTypeColor4',
    15: 'kRendererObjectValueTypeEnum',
    4: 'kRendererObjectValueTypeFloat',
    1: 'kRendererObjectValueTypeInt',
    19: 'kRendererObjectValueTypeLocation',
    14: 'kRendererObjectValueTypeMatrix',
    16: 'kRendererObjectValueTypeNormal',
    10: 'kRendererObjectValueTypePoint2',
    11: 'kRendererObjectValueTypePoint3',
    12: 'kRendererObjectValueTypePoint4',
    17: 'kRendererObjectValueTypePointer',
    19: 'kRendererObjectValueTypeShader',
    13: 'kRendererObjectValueTypeString',
    2: 'kRendererObjectValueTypeUint',
    -1: 'kRendererObjectValueTypeUnknown',
    7: 'kRendererObjectValueTypeVector2',
    8: 'kRendererObjectValueTypeVector3',
    9: 'kRendererObjectValueTypeVector4',
    'color': 'kRendererOutputTypeColor',
    'deep': 'kRendererOutputTypeDeep',
    'none': 'kRendererOutputTypeForceNone',
    'merge': 'kRendererOutputTypeMerge',
    'prescript': 'kRendererOutputTypePreScript',
    'raw': 'kRendererOutputTypeRaw',
    'script': 'kRendererOutputTypeScript',
    'shadow': 'kRendererOutputTypeShadow shadow',
    'renderMethodType': 'kTerminalOpStateArgRenderMethodType',
    'system': 'kTerminalOpStateArgSystem'
}

_USD_TYPE_MAP_ = {
    RenderingAPI.RendererInfo.kRendererObjectValueTypeString: Sdf.ValueTypeNames.String,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeInt: Sdf.ValueTypeNames.Int,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeFloat: Sdf.ValueTypeNames.Float,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeColor3: Sdf.ValueTypeNames.Color3f,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeVector3: Sdf.ValueTypeNames.Float3,
    RenderingAPI.RendererInfo.kRendererObjectValueTypeNormal: Sdf.ValueTypeNames.Float3,
    RenderingAPI.RendererInfo.kRendererObjectValueTypePointer: Sdf.ValueTypeNames.String
}
