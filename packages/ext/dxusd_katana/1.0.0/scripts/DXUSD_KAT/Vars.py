import os, re

from fnpxr import Sdf
VTN = Sdf.ValueTypeNames

import DXRulebook as rb

# path seperator
SEP  = os.path.sep
_SEP = '.' + SEP

T = rb.Tags('USD')
D = rb.Coder('D', 'USD', T.PUB3)
F = rb.Coder('F', 'USD', T.PUB3)
N = rb.Coder('N', 'USD', T.PUB3)

#-------------------------------------------------------------------------------
PXRTLUX = {
    'PxrDomeLight': 'DomeLight',
    'PxrRectLight': 'RectLight',
    'PxrDiskLight': 'DiskLight',
    'PxrDistantLight': 'DistantLight',
    'PxrSphereLight': 'SphereLight',
    'PxrCylinderLight': 'CylinderLight',
    'PxrEnvDayLight': 'PxrEnvDayLight',
    'PxrAovLight': 'PxrAovLight'
}

#-------------------------------------------------------------------------------
_Light = {
    'intensity': ('intensity', VTN.Float),
    'exposure': ('exposure', VTN.Float),
    'diffuse': ('diffuse', VTN.Float),
    'specular': ('specular', VTN.Float),
    'areaNormalize': ('normalize', VTN.Bool),
    'lightColor': ('color', VTN.Color3f),
    'enableTemperature': ('enableColorTemperature', VTN.Bool),
    'temperature': ('colorTemperature', VTN.Float),
}

_ShapingAPI = {
    'emissionFocus': ('shaping:focus', VTN.Float),
    'emissionFocusTint': ('shaping:focusTint', VTN.Color3f),
    'coneAngle': ('shaping:cone:angle', VTN.Float),
    'coneSoftness': ('shaping:cone:softness', VTN.Float),
    'iesProfile': ('shaping:ies:file', VTN.Asset),
    'iesProfileScale': ('shaping:ies:angleScale', VTN.Float),
    'iesProfileNormalize': ('shaping:ies:normalize', VTN.Bool),
}

_ShadowAPI = {
    'enableShadows': ('shadow:enable', VTN.Bool),
    'shadowColor': ('shadow:color', VTN.Color3f),
    'shadowDistance': ('shadow:distance', VTN.Float),
    'shadowFalloff': ('shadow:falloff', VTN.Float),
    'shadowFalloffGamma': ('shadow:falloffGamma', VTN.Float),
}

_RiLightAPI = {
    'fixedSampleCount': ('ri:sampling:fixedSampleCount', VTN.Int),
    'importanceMultiplier': ('ri:sampling:importanceMultiplier', VTN.Float),
    'intensityNearDist': ('ri:intensityNearDist', VTN.Float),
    'lightGroup': ('ri:lightGroup', VTN.String),
    'thinShadow': ('ri:shadow:thinShadow', VTN.Bool),
    'traceLightPaths': ('ri:trace:lightPaths', VTN.Bool),
}

_RiTextureAPI = {
    'lightColorMap': ('texture:file', VTN.Asset),
    'colorMapGamma': ('ri:light:colorMapGamma', VTN.Vector3f),
    'colorMapSaturation': ('ri:light:colorMapSaturation', VTN.Float),
}

#-------------------------------------------------------------------------------
_PxrEnvDayLight = {
    'day': ('day', VTN.Int),
    'haziness': ('haziness', VTN.Float),
    'hour': ('hour', VTN.Float),
    'latitude': ('latitude', VTN.Float),
    'longitude': ('longitude', VTN.Float),
    'month': ('month', VTN.Int),
    'skyTint': ('skyTint', VTN.Color3f),
    'sunDirection': ('sunDirection', VTN.Vector3f),
    'sunSize': ('sunSize', VTN.Float),
    'sunTint': ('sunTint', VTN.Color3f),
    'year': ('year', VTN.Int),
    'zone': ('zone', VTN.Float),
}

_PxrAovLight = {
    'aovName': ('aovName', VTN.String),
    'inPrimaryHit': ('inPrimaryHit', VTN.Bool),
    'inReflection': ('inReflection', VTN.Bool),
    'inRefraction': ('inRefraction', VTN.Bool),
    'invert': ('invert', VTN.Bool),
    'onVolumeBoundaries': ('onVolumeBoundaries', VTN.Bool),
    'useColor': ('useColor', VTN.Bool),
    'useThroughput': ('useThroughput', VTN.Bool),
}

#-------------------------------------------------------------------------------
_LightFilterAPI = {
    'combineMode': ('ri:combineMode', VTN.Token),
    'density': ('ri:density', VTN.Float),
    'invert': ('ri:invert', VTN.Bool),
    'intensity': ('ri:intensity', VTN.Float),
    'exposure': ('ri:exposure', VTN.Float),
    'diffuse': ('ri:diffuse', VTN.Float),
    'specular': ('ri:specular', VTN.Float),
}

_PxrIntMultLightFilter = {
    'saturation': ('color:saturation', VTN.Float),
}

_PxrBarnLightFilter = {
    'barnMode': ('barnMode', VTN.Token),    # physical 0, analytic 1
    'width': ('widht', VTN.Float),
    'height': ('height', VTN.Float),
    'radius': ('radius', VTN.Float),
    'directional': ('analytic:directional', VTN.Bool),
    'shearX': ('analytic:shearX', VTN.Float),
    'shearY': ('analytic:shearY', VTN.Float),
    'apex': ('analytic:apex', VTN.Float),
    'useLightDirection': ('analytic:useLightDirection', VTN.Bool),
    'densityNear': ('analytic:density:nearDistance', VTN.Float),
    'densityFar': ('analytic:density:farDistance', VTN.Float),
    'densityNearVal': ('analytic:density:nearValue', VTN.Float),
    'densityFarVal': ('analytic:density:farValue', VTN.Float),
    'densityPow': ('analytic:density:exponent', VTN.Float),
    'edge': ('edgeThickness', VTN.Float),
    'preBarn': ('preBarnEffect', VTN.Token),    # noEffect 0, cone 1, noLight 2
    'scaleWidth': ('scale:width', VTN.Float),
    'scaleHeight': ('scale:height', VTN.Float),
    'top': ('refine:top', VTN.Float),
    'bottom': ('refine:bottom', VTN.Float),
    'left': ('refine:left', VTN.Float),
    'right': ('refine:right', VTN.Float),
    'topEdge': ('edgeScale:top', VTN.Float),
    'bottomEdge': ('edgeScale:bottom', VTN.Float),
    'leftEdge': ('edgeScale:left', VTN.Float),
    'rightEdge': ('edgeScale:right', VTN.Float),
}

_PxrCookieLightFilter = {
    'cookieMode': ('cookieMode', VTN.Token),    # physical 0, analytic 1
    'width': ('width', VTN.Float),
    'height': ('height', VTN.Float),
    'map': ('texture:map', VTN.Asset),
    'tileMode': ('texture:wrapMode', VTN.Token),    # off 0, repeat 2, clamp 1
    'fillColor': ('texture:fillColor', VTN.Color3f),
    'premultipliedAlpha': ('texture:premultipliedAlpha', VTN.Bool),
    'invertU': ('texture:invertU', VTN.Bool),
    'invertV': ('texture:invertV', VTN.Bool),
    'scaleU': ('texture:scaleU', VTN.Float),
    'scaleV': ('texture:scaleV', VTN.Float),
    'offsetU': ('texture:offsetU', VTN.Float),
    'offsetV': ('texture:offsetV', VTN.Float),
    'directional': ('analytic:directional', VTN.Bool),
    'shearX': ('analytic:shearX', VTN.Float),
    'shearY': ('analytic:shearY', VTN.Float),
    'apex': ('analytic:apex', VTN.Float),
    'useLightDirection': ('analytic:useLightDirection', VTN.Bool),
    'blur': ('analytic:blur:amount', VTN.Float),
    'sBlurMult': ('analytic:blur:sMult', VTN.Float),
    'tBlurMult': ('analytic:blur:tMult', VTN.Float),
    'blurNearDist': ('analytic:blur:nearDistance', VTN.Float),
    'blurMidpoint': ('analytic:blur:midpoint', VTN.Float),
    'blurFarDist': ('analytic:blur:farDistance', VTN.Float),
    'blurNearVal': ('analytic:blur:nearValue', VTN.Float),
    'blurMidVal': ('analytic:blur:midValue', VTN.Float),
    'blurFarVal': ('analytic:blur:farValue', VTN.Float),
    'blurPow': ('analytic:blur:exponent', VTN.Float),
    'densityNearDist': ('analytic:density:nearDistance', VTN.Float),
    'densityMidpoint': ('analytic:density:midpoint', VTN.Float),
    'densityFarDist': ('analytic:density:farDistance', VTN.Float),
    'densityNearVal': ('analytic:density:nearValue', VTN.Float),
    'densityMidVal': ('analytic:density:midValue', VTN.Float),
    'densityFarVal': ('analytic:density:farValue', VTN.Float),
    'densityPow': ('analytic:density:exponent', VTN.Float),
    'saturation': ('color:saturation', VTN.Float),
    'midpoint': ('color:midpoint', VTN.Float),
    'contrast': ('color:contrast', VTN.Float),
    'whitepoint': ('color:whitepoint', VTN.Float),
    'tint': ('color:tint', VTN.Color3f),
}

_PxrRampLightFilter = {
    'rampMode': ('rampMode', VTN.Int),
    'beginDist': ('beginDistance', ),
    'endDist': ('endDistance', ),
    'falloff': ('falloff', VTN.Int),
    # 'falloff_Knots': ('falloff:knots', VTN.FloatArray),
    # 'falloff_Floats': ('falloff:floats', VTN.FloatArray),
    # 'falloff_Interpolation': ('falloff:interpolation', VTN.Token),
    'falloff_Knots': ('falloffRamp:spline:positions', VTN.FloatArray),
    'falloff_Floats': ('falloffRamp:spline:values', VTN.FloatArray),
    'falloff_Interpolation': ('falloffRamp:spline:interpolation', VTN.Token),
    'colorRamp': ('colorRamp', VTN.Int),
    # 'colorRamp_Knots': ('colorRamp:knots', VTN.FloatArray),
    # 'colorRamp_Colors': ('colorRamp:colors', VTN.Color3fArray),
    # 'colorRamp_Interpolation': ('colorRamp:interpolation', VTN.Token),
    'colorRamp_Knots': ('colorRamp:spline:positions', VTN.FloatArray),
    'colorRamp_Colors': ('colorRamp:spline:values', VTN.Color3fArray),
    'colorRamp_Interpolation': ('colorRamp:spline:interpolation', VTN.Token),
}

_PxrRodLightFilter = {
    'width': ('width', VTN.Float),
    'height': ('height', VTN.Float),
    'depth': ('depth', VTN.Float),
    'radius': ('radius', VTN.Float),
    'edge': ('edgeThickness', VTN.Float),
    'scaleWidth': ('scale:width', VTN.Float),
    'scaleHeight': ('scale:height', VTN.Float),
    'scaleDepth': ('scale:depth', VTN.Float),
    'top': ('refine:top', VTN.Float),
    'bottom': ('refine:bottom', VTN.Float),
    'left': ('refine:left', VTN.Float),
    'right': ('refine:right', VTN.Float),
    'front': ('refine:front', VTN.Float),
    'back': ('refine:back', VTN.Float),
    'topEdge': ('edgeScale:top', VTN.Float),
    'bottomEdge': ('edgeScale:bottom', VTN.Float),
    'leftEdge': ('edgeScale:left', VTN.Float),
    'rightEdge': ('edgeScale:right', VTN.Float),
    'frontEdge': ('edgeScale:front', VTN.Float),
    'backEdge': ('edgeScale:back', VTN.Float),
    'saturation': ('color:saturation', VTN.Float),
    'falloff': ('falloff', VTN.Int),
    # 'falloff_Knots': ('falloff:knots', VTN.FloatArray),
    # 'falloff_Floats': ('falloff:floats', VTN.FloatArray),
    # 'falloff_Interpolation': ('falloff:interpolation', VTN.Token),
    'falloff_Knots': ('falloffRamp:spline:positions', VTN.FloatArray),
    'falloff_Floats': ('falloffRamp:spline:values', VTN.FloatArray),
    'falloff_Interpolation': ('falloffRamp:spline:interpolation', VTN.Token),
    'colorRamp': ('colorRamp', VTN.Int),
    # 'colorRamp_Knots': ('colorRamp:knots', VTN.FloatArray),
    # 'colorRamp_Colors': ('colorRamp:colors', VTN.Color3fArray),
    # 'colorRamp_Interpolation': ('colorRamp:interpolation', VTN.Token),
    'colorRamp_Knots': ('colorRamp:spline:positions', VTN.FloatArray),
    'colorRamp_Colors': ('colorRamp:spline:values', VTN.Color3fArray),
    'colorRamp_Interpolation': ('colorRamp:spline:interpolation', VTN.Token),
}
