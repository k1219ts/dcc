import os
from python import stamp_info

nuke.pluginAddPath('./python')
nuke.pluginAddPath('./icons')
nuke.pluginAddPath('./gizmos')
nuke.pluginAddPath('./plugins')
nuke.pluginAddPath('./plugins/cryptomatte')
nuke.pluginAddPath('./plugins/WaveMachine')
nuke.pluginAddPath('./toolset')

if 'SHOW' in os.environ:
    nuke.pluginAddPath('/show/{SHOW}/_config/nuke'.format(SHOW=os.environ['SHOW']))

# nuke.ViewerProcess.register("AlexaV3Rec709", nuke.createNode, ("Vectorfield","vfield_file /backstage/apps/Nuke/Globals/Lookup/AlexaV3_EI0800_LogC2Video_Rec709_EE_nuke3d.cube colorspaceIn AlexaV3LogC"))
# nuke.ViewerProcess.register("MRZ", nuke.createNode, ("Vectorfield","vfield_file /backstage/apps/Nuke/Globals/Lookup/AlexaV3_K1S1_LogC2Video_Rec709_EE_nuke3d.cube colorspaceIn AlexaV3LogC"))
# nuke.ViewerProcess.register("GCD1", nuke.createNode, ("Vectorfield","vfield_file /show/gcd1/asset/LUT/170414_LUT/RedColor3RedLogFilm_to_P7V2_D65_G22.cube colorspaceIn Cineon"))
# nuke.ViewerProcess.register("GOD", nuke.createNode, ("Vectorfield","vfield_file /backstage/apps/Nuke/Globals/Lookup/AlexaV3_K1S1_LogC2Video_Rec709_EE_nuke3d.cube colorspaceIn AlexaV3LogC"))
# nuke.ViewerProcess.register("SSR", nuke.createNode, ("Vectorfield","vfield_file /stuff/ssr/stuff/LUT/20180521/P3_DCI_2.6Gamma.cub colorspaceIn AlexaV3LogC colorspaceOut linear"))
# nuke.ViewerProcess.register("ASD", nuke.createNode, ("Vectorfield","vfield_file /backstage/apps/Nuke/Globals/Lookup/AlexaV3_K1S1_LogC2Video_Rec709_EE_nuke3d.cube colorspaceIn AlexaV3LogC colorspaceOut linear"))
# nuke.ViewerProcess.register("HOLO", nuke.createNode, ("Vectorfield","vfield_file /stuff/holo/stuff/lut/holo_lut.cube colorspaceIn Cineon"))
# nuke.ViewerProcess.register("MGD", nuke.createNode, ("Vectorfield","vfield_file /stuff/mgd/stuff/LUT/20200311_VFX_RWG_Log3G10_to_REC709.cube colorspaceIn REDLog colorspaceOut linear"))

nuke.root().knob('luts').addCurve("cineon2", "curve l 0 x0.4156930149 0.1223881245 x0.6968169212 0.7022315264 x0.7599790096 0.821751833 k x1 1.285815954 s1.744879961 t1.700000048")

nuke.knobDefault('Text.font' , os.environ['REZ_NUKE_BASE'] + '/scripts/fonts/batang.ttf')

#Adding custom format
nuke.addFormat('1280 720 Half_HD')
nuke.addFormat('640 360 Half_HD_Proxy')
nuke.addFormat('960 540 HD_Proxy')
nuke.addFormat('2200 1100 WEST')
nuke.addFormat('2048 1318 GOD')
nuke.addFormat('2048 872 GOD_Out')
nuke.addFormat('2185 1152 GCD1')
nuke.addFormat('2240 1152 GCD2')
nuke.addFormat('2048 858 DOK')
nuke.addFormat('1998 1286 PMC')
nuke.addFormat('1998 1080 PMC_out')
nuke.addFormat('2048 1152 IMT')
nuke.addFormat('2048 1426 BDS')
nuke.addFormat('2276 1200 MKK3')
nuke.addFormat('3840 2024 HOLO')
nuke.addFormat('1920 960 DIE')

# ASSIGN KNOB DEFAULTS
nuke.knobDefault("Blur.label", "[value size]")
nuke.knobDefault("Blur.channels", "rgba")
nuke.knobDefault("Roto.output", "rgba")
nuke.knobDefault("Root.format", "GOD")
# nuke.knobDefault("Hit_custom.format", "SSY")
nuke.knobDefault('AddTimeCode.startcode', '00:00:00:11')

nuke.knobDefault("Defocus.channels", "rgba")
nuke.knobDefault("Read.label", "[value first]-[value last]")
nuke.knobDefault("AppendClip.label", "[knob firstFrame] - [knob lastFrame]")
nuke.knobDefault("Multiply.channels", "rgba")
nuke.knobDefault("FrameRange.label", "[knob first_frame]-[knob last_frame]([expr [knob last_frame]-[knob first_frame]+1])")
nuke.knobDefault("Retime.label", "[knob output.first] - [knob output.last]([expr [knob output.last]-[knob output.first]+1])")
nuke.knobDefault("LayerContactSheet.showLayerNames", "True")
nuke.knobDefault("VectorBlur.uv", "motion")

#useGPUIfAvailable set Expression
#nuke.knobDefault('useGPUIfAvailable', '{$gui}')
nuke.knobDefault('useGPUIfAvailable','false')
