#coding:utf-8
import opentimelineio as otio
import xlrd2
import Msg
from Define import Column
from adapters.calculator import *

metadata = {
    "fcp_xml": {
        "effectcategory": "Text",
        "effectid": "Text",
        "effecttype": "generator",
        "mediatype": "video",
        "parameter" : [
        {
            "name": "Text",
            "parameterid": "str",
            "value": "MON_0010"
        },
        {
            "name": "Font",
            "parameterid": "fontname",
            "value": "Lucida Grande"
        },
        {
            "name": "Size",
            "parameterid": "fontsize",
            "value": "36",
            "valuemax": "1000",
            "valuemin": "0"
        },
        {
            "name": "Style",
            "parameterid": "fontstyle",
            "value": "1",
            "valuelist": {
                "valueentry": [
                    {
                        "name": "Plain",
                        "value": "1"
                    },
                    {
                        "name": "Bold",
                        "value": "2"
                    },
                    {
                        "name": "Italic",
                        "value": "3"
                    },
                    {
                        "name": "Bold/Italic",
                        "value": "4"
                    }
                ]
            },
            "valuemax": "4",
            "valuemin": "1"
        },
        {
            "name": "Alignment",
            "parameterid": "fontalign",
            "value": "2",
            "valuelist": {
                "valueentry": [
                    {
                        "name": "Left",
                        "value": "1"
                    },
                    {
                        "name": "Center",
                        "value": "2"
                    },
                    {
                        "name": "Right",
                        "value": "3"
                    }
                ]
            },
            "valuemax": "3",
            "valuemin": "1"
        },
        {
            "name": "Font Color",
            "parameterid": "fontcolor",
            "value": {
                "alpha": "255",
                "blue": "0",
                "green": "0",
                "red": "255"
            }
        },
        {
            "name": "Origin",
            "parameterid": "origin",
            "value": {
                "horiz": "0",
                "vert": "0"
            }
        },
        {
            "name": "Tracking",
            "parameterid": "fonttrack",
            "value": "1",
            "valuemax": "200",
            "valuemin": "-200"
        },
        {
            "name": "Leading",
            "parameterid": "leading",
            "value": "0",
            "valuemax": "100",
            "valuemin": "-100"
        },
        {
            "name": "Aspect",
            "parameterid": "aspect",
            "value": "1",
            "valuemax": "5",
            "valuemin": "0.1"
        },
        {
            "name": "Auto Kerning",
            "parameterid": "autokern",
            "value": "TRUE"
        },
        {
            "name": "Use Subpixel",
            "parameterid": "subpixel",
            "value": "TRUE"
        }
    ]
    }
}

clipmetadata = {
    "filter": [
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "basic",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Basic Motion",
                        "parameter": [
                            {
                                "name": "Scale",
                                "parameterid": "scale",
                                "value": "75",
                                "valuemax": "1000",
                                "valuemin": "0"
                            },
                            {
                                "name": "Rotation",
                                "parameterid": "rotation",
                                "value": "0",
                                "valuemax": "8640",
                                "valuemin": "-8640"
                            },
                            {
                                "name": "Center",
                                "parameterid": "center",
                                "value": {
                                    "horiz": "0.379167",
                                    "vert": "-0.400926"
                                }
                            },
                            {
                                "name": "Anchor Point",
                                "parameterid": "centerOffset",
                                "value": {
                                    "horiz": "0",
                                    "vert": "0"
                                }
                            }
                        ]
                    }
                },
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "dropshadow",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Drop Shadow",
                        "parameter": [
                            {
                                "name": "offset",
                                "parameterid": "offset",
                                "value": "2",
                                "valuemax": "100",
                                "valuemin": "-100"
                            },
                            {
                                "name": "angle",
                                "parameterid": "angle",
                                "value": "135",
                                "valuemax": "720",
                                "valuemin": "-720"
                            },
                            {
                                "name": "color",
                                "parameterid": "color",
                                "value": {
                                    "alpha": "0",
                                    "blue": "0",
                                    "green": "0",
                                    "red": "0"
                                }
                            },
                            {
                                "name": "softness",
                                "parameterid": "softness",
                                "value": "10",
                                "valuemax": "100",
                                "valuemin": "0"
                            },
                            {
                                "name": "opacity",
                                "parameterid": "opacity",
                                "value": "50",
                                "valuemax": "100",
                                "valuemin": "0"
                            }
                        ]
                    },
                    "enabled": "FALSE"
                },
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "motionblur",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Motion Blur",
                        "parameter": [
                            {
                                "name": "% Blur",
                                "parameterid": "duration",
                                "value": "500",
                                "valuemax": "1000",
                                "valuemin": "0"
                            },
                            {
                                "name": "Samples",
                                "parameterid": "samples",
                                "value": "4",
                                "valuemax": "16",
                                "valuemin": "1"
                            }
                        ]
                    },
                    "enabled": "FALSE"
                },
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "crop",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Crop",
                        "parameter": [
                            {
                                "name": "left",
                                "parameterid": "left",
                                "value": "0",
                                "valuemax": "100",
                                "valuemin": "0"
                            },
                            {
                                "name": "right",
                                "parameterid": "right",
                                "value": "0",
                                "valuemax": "100",
                                "valuemin": "0"
                            },
                            {
                                "name": "top",
                                "parameterid": "top",
                                "value": "0",
                                "valuemax": "100",
                                "valuemin": "0"
                            },
                            {
                                "name": "bottom",
                                "parameterid": "bottom",
                                "value": "0",
                                "valuemax": "100",
                                "valuemin": "0"
                            },
                            {
                                "name": "edgefeather",
                                "parameterid": "edgefeather",
                                "value": "0",
                                "valuemax": "100",
                                "valuemin": "0"
                            }
                        ]
                    }
                },
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "deformation",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Distort",
                        "parameter": [
                            {
                                "name": "Upper Left",
                                "parameterid": "ulcorner",
                                "value": {
                                    "horiz": "-0.5",
                                    "vert": "-0.5"
                                }
                            },
                            {
                                "name": "Upper Right",
                                "parameterid": "urcorner",
                                "value": {
                                    "horiz": "0.5",
                                    "vert": "-0.5"
                                }
                            },
                            {
                                "name": "Lower Right",
                                "parameterid": "lrcorner",
                                "value": {
                                    "horiz": "0.5",
                                    "vert": "0.5"
                                }
                            },
                            {
                                "name": "Lower Left",
                                "parameterid": "llcorner",
                                "value": {
                                    "horiz": "-0.5",
                                    "vert": "0.5"
                                }
                            },
                            {
                                "name": "Aspect",
                                "parameterid": "aspect",
                                "value": "0",
                                "valuemax": "10000",
                                "valuemin": "-10000"
                            }
                        ]
                    }
                },
                {
                    "effect": {
                        "effectcategory": "motion",
                        "effectid": "opacity",
                        "effecttype": "motion",
                        "mediatype": "video",
                        "name": "Opacity",
                        "parameter": {
                            "name": "opacity",
                            "parameterid": "opacity",
                            "value": "100",
                            "valuemax": "100",
                            "valuemin": "0"
                        }
                    }
                }
    ]
}
effects = [otio.schema.Effect(name = "Basic Motion", metadata={"fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "basic",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": [
                        {
                            "name": "Scale",
                            "parameterid": "scale",
                            "value": "75",
                            "valuemax": "1000",
                            "valuemin": "0"
                        },
                        {
                            "name": "Rotation",
                            "parameterid": "rotation",
                            "value": "0",
                            "valuemax": "8640",
                            "valuemin": "-8640"
                        },
                        {
                            "name": "Center",
                            "parameterid": "center",
                            "value": {
                                "horiz": "0.379167",
                                "vert": "-0.400926"
                            }
                        },
                        {
                            "name": "Anchor Point",
                            "parameterid": "centerOffset",
                            "value": {
                                "horiz": "0",
                                "vert": "0"
                            }
                        }
                    ]
                }
            }),
            otio.schema.Effect(name = "Drop Shadow", metadata={"fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "dropshadow",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": [
                        {
                            "name": "offset",
                            "parameterid": "offset",
                            "value": "2",
                            "valuemax": "100",
                            "valuemin": "-100"
                        },
                        {
                            "name": "angle",
                            "parameterid": "angle",
                            "value": "135",
                            "valuemax": "720",
                            "valuemin": "-720"
                        },
                        {
                            "name": "color",
                            "parameterid": "color",
                            "value": {
                                "alpha": "0",
                                "blue": "0",
                                "green": "0",
                                "red": "0"
                            }
                        },
                        {
                            "name": "softness",
                            "parameterid": "softness",
                            "value": "10",
                            "valuemax": "100",
                            "valuemin": "0"
                        },
                        {
                            "name": "opacity",
                            "parameterid": "opacity",
                            "value": "50",
                            "valuemax": "100",
                            "valuemin": "0"
                        }
                    ]
                }}),
           otio.schema.Effect(name = "Motion Blur", metadata={"fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "motionblur",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": [
                        {
                            "name": "% Blur",
                            "parameterid": "duration",
                            "value": "500",
                            "valuemax": "1000",
                            "valuemin": "0"
                        },
                        {
                            "name": "Samples",
                            "parameterid": "samples",
                            "value": "4",
                            "valuemax": "16",
                            "valuemin": "1"
                        }
                    ]
                }}),
           otio.schema.Effect(name="Crop", metadata={
                "fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "crop",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": [
                        {
                            "name": "left",
                            "parameterid": "left",
                            "value": "0",
                            "valuemax": "100",
                            "valuemin": "0"
                        },
                        {
                            "name": "right",
                            "parameterid": "right",
                            "value": "0",
                            "valuemax": "100",
                            "valuemin": "0"
                        },
                        {
                            "name": "top",
                            "parameterid": "top",
                            "value": "0",
                            "valuemax": "100",
                            "valuemin": "0"
                        },
                        {
                            "name": "bottom",
                            "parameterid": "bottom",
                            "value": "0",
                            "valuemax": "100",
                            "valuemin": "0"
                        },
                        {
                            "name": "edgefeather",
                            "parameterid": "edgefeather",
                            "value": "0",
                            "valuemax": "100",
                            "valuemin": "0"
                        }
                    ]
                }
            }),
           otio.schema.Effect(name="Distort", metadata={
                "fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "deformation",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": [
                        {
                            "name": "Upper Left",
                            "parameterid": "ulcorner",
                            "value": {
                                "horiz": "-0.5",
                                "vert": "-0.5"
                            }
                        },
                        {
                            "name": "Upper Right",
                            "parameterid": "urcorner",
                            "value": {
                                "horiz": "0.5",
                                "vert": "-0.5"
                            }
                        },
                        {
                            "name": "Lower Right",
                            "parameterid": "lrcorner",
                            "value": {
                                "horiz": "0.5",
                                "vert": "0.5"
                            }
                        },
                        {
                            "name": "Lower Left",
                            "parameterid": "llcorner",
                            "value": {
                                "horiz": "-0.5",
                                "vert": "0.5"
                            }
                        },
                        {
                            "name": "Aspect",
                            "parameterid": "aspect",
                            "value": "0",
                            "valuemax": "10000",
                            "valuemin": "-10000"
                        }
                    ]
                }
            }),
            otio.schema.Effect(name='Opacity', metadata={
                "fcp_xml": {
                    "effectcategory": "motion",
                    "effectid": "opacity",
                    "effecttype": "motion",
                    "mediatype": "video",
                    "parameter": {
                        "name": "opacity",
                        "parameterid": "opacity",
                        "value": "100",
                        "valuemax": "100",
                        "valuemin": "0"
                    }
                }
            })
           ]

# xmlFilePath = '/prod_nas/__DD_PROD/SLC/edit/20201116/S39_DAT_locationEdit/Kang_action_1114_toEditor_1.xml'
# xmlFilePath = '/prod_nas/__DD_PROD/SLC/edit/20201116/S39_DAT_locationEdit/Kang_action_1114_toEditor.xml'
xmlFilePath = '/prod_nas/__DD_PROD/EMD/edit/20201130_CAR_OFF_RAT/201130_preCG/XML/Emergency_A01_PreCG_Guide_RAT_EDL_201130.xml'
# xmlFilePath = '/prod_nas/__DD_PROD/EMD/edit/20201130/201130_preCG/XML/Emergency_A01_PreCG_Guide_OFF_EDL_201130.xml'
# xmlFilePath = '/prod_nas/__DD_PROD/EMD/new/edit/20201130/201130_preCG/XML/Emergency_A01_PreCG_Guide_CAR_EDL_201130.xml'
excelFilePath = xmlFilePath.replace('.xml', '.xls')

otioData = otio.adapters.read_from_file(xmlFilePath, 'fcp_xml')
globalStartFrame = otioData.global_start_time.to_frames()
globalDuration = otioData.duration()
# print globalDuration

burnInIndex = 0
for index, i in enumerate(otioData.video_tracks()):
    if len(i) == 0:
        burnInIndex = index
        break
else:
    burnInIndex = len(otioData.video_tracks())
    otioData.tracks.append(otio.schema.Track())
    print len(otioData.video_tracks())

cleanupTimeline = cleanupTrackData(otioData)

book = xlrd2.open_workbook(excelFilePath)
sheet = book.sheet_by_name('scan_list')

cutIndex = 0
excelRowIndex = 1
editCutInFrameList = sorted(cleanupTimeline.keys())
while cutIndex < len(cleanupTimeline):
    if cutIndex == 0:
        data = sheet.row_values(excelRowIndex)
        cutInGap = data[Column.MOV_CUT_IN.value] - globalStartFrame
        print cutInGap
        if cutInGap != 0:
            startTime = otio.opentime.RationalTime(0, 24)
            durationTime = otio.opentime.RationalTime(cutInGap, 24)
            gapSchema = otio.schema.Gap(source_range=otio.opentime.TimeRange(startTime, durationTime))
            otioData.video_tracks()[burnInIndex].append(gapSchema)

    while 'main' not in sheet.row_values(excelRowIndex)[Column.TYPE.value]:
        excelRowIndex += 1

    data = sheet.row_values(excelRowIndex)
    shotName = data[Column.SHOT_NAME.value]
    cutIn = data[Column.MOV_CUT_IN.value] - globalStartFrame
    # print cutIn, editCutInFrameList[cutIndex]
    while int(cutIn) != editCutInFrameList[cutIndex]:
        duration = editCutInFrameList[cutIndex + 1] - editCutInFrameList[cutIndex]
        startIn = editCutInFrameList[cutIndex]

        startTime = otio.opentime.RationalTime(startIn, 24)
        durationTime = otio.opentime.RationalTime(duration, 24)
        gapSchema = otio.schema.Gap(source_range=otio.opentime.TimeRange(startTime, durationTime))
        otioData.video_tracks()[burnInIndex].append(gapSchema)

        cutIndex += 1

    # print shotName, cutIn, editCutInFrameList[cutIndex]
    try:
        clip = cleanupTimeline[editCutInFrameList[cutIndex]]['clip'][0]
    except:
        duration = editCutInFrameList[cutIndex + 1] - editCutInFrameList[cutIndex]
        startIn = editCutInFrameList[cutIndex]

        startTime = otio.opentime.RationalTime(startIn, 24)
        durationTime = otio.opentime.RationalTime(duration, 24)
        gapSchema = otio.schema.Gap(source_range=otio.opentime.TimeRange(startTime, durationTime))
        otioData.video_tracks()[burnInIndex].append(gapSchema)

        cutIndex += 1
        excelRowIndex += 1
        continue
    try:
        duration = editCutInFrameList[cutIndex + 1] - editCutInFrameList[cutIndex]
    except:
        duration = globalDuration.to_frames() - editCutInFrameList[cutIndex]
    startIn = editCutInFrameList[cutIndex]

    startTime = otio.opentime.RationalTime(startIn, 24)
    durationTime = otio.opentime.RationalTime(duration, 24)

    metadata['fcp_xml']['parameter'][0]['value'] = shotName
    mediaReference = otio.schema.GeneratorReference(name='Text', metadata=metadata)

    burnInClip = otio.schema.Clip(name=shotName, source_range=otio.opentime.TimeRange(startTime, durationTime),
                                  media_reference=mediaReference, metadata=clipmetadata)
    for effect in effects:
        burnInClip.effects.append(effect)

    otioData.video_tracks()[burnInIndex].append(burnInClip)

    cutIndex += 1
    excelRowIndex += 1

# print dir(otioData.tracks)# .video_tracks()[burnInIndex]
# for index, i in enumerate(otioData.video_tracks()):
#     if len(i) == 0:
#         print index

otioData.tracks[:] = otioData.video_tracks()

burnInXml = xmlFilePath.replace('.xml', '_burnin.xml')
otio.adapters.write_to_file(otioData, burnInXml)