========================================================================================================
```
unique dependency node ID (0x80000~0xfffff: 524288 ids)

The numeric ID ranges have been divided like this. Ids in range

0 - 0x0007FFFF has been reserved for plug-ins that will forever be internal to your site.
But while we reserved those for your internal usage,
I would recommend having your own range just in case you share a file and a plug-in with a customer,
vendor or contractor many years later.

0x00080000 - 0x000FFFFF has been reserved for Maya devkit samples.
If you customize one of these plug-in examples, you should change the id to avoid future conflicts.

0x00000000 ~ 0x0007FFFF: internal use
0x00080000 ~ 0x000FFFFF: devkit examples
0x00100FFF ~ 0x00FFFFFF: global IDs
```
========================================================================================================

# MAYA
### maya
```
dxComponent         0x10170001, 0x10170002
dxAssembly          0x10170003, 0x10170004
```
### maya_animation
```
dxIk                0x00124843
ghProjectMesh       0x87668
```
### maya_layout
```
ZNumToString        0x93001
```
### maya_rigging
```
curveLength2ParamU  0x8704
fexBlendShape       0x00000002
pyInbetweenShape    0x10400000

```
### DXUSD_MAYA
```
dxBlock             0x10170010, 0x10170011
dxTimeOffset        0x10170020
dxRig               0x79000, 0x79001
dxCamera            0x79002, 0x79003
```
# RenderMan
```
Please note: Pixar has reserved maya node ids:

0x00101261-0x001012ff (1,053,281 - 1,053,439)

for our rfm templates.

Customers should request node id blocks from Autodesk to prevent
compatibility issues.  Sites should further adopt naming conventions
in the form of four-letter prefixes to reduce the risk of template
collision. The prefixes px and rman are reserved by Pixar.

rfmManifold.rslt:   0x00101261-0x00101269 (1,053,281-1,053,289)

rfmTexture2d.rslt:  0x0010126A-0x0010127D (1,053,290-1,053,309)

rfmTexture3d.rslt:  0x0010127E-0x00101291 (1,053,310-1,053,329)

rfmUtility.rslt:    0x00101292-0x001012AF (1,053,330-1,053,359)

plausible nodes:    0x001012B0-0x001012C3 (1,053,360-1,053,399)
```
```
// DXR-REYES
dxrTexture              2100001
dxrZelosOceanWave       2100101
dxrZelosTwoOceanWave    2100102
dxrZelosDoubleOceanWave 2100103

// DX-RIS
DxTexture               2000001
DxCustomAOV             2000004
DxGrade                 2000005
DxHair                  2000006
DxLinearize             2000007
DxID                    2000009
DxWire                  2000010
DxShadingContext        2000011
DxOBJ                   2000012
DxPrimvar               2000013
DxCrowdID               2000014
DxSideMask              2000015
ZOceanWavePattern       2000101
ZTwoOceanWavePattern    2000102
DxConstant              2000021
DxFoam                  2000023
DxObjectAOV             2000024
DxZFXOceanLoader        2000025
DxManifold2D            2000026
ZVOceanLoader           2000027
PxrLightProbe           2000028
DxFisheye               2000029
DxSphere                2000030
PxrBoraOcean            2000031

DxOmniDirectionalStereo 2000032
DxCroppedFisheye        2000033
DxBeyondScreen          2000034

DxBoraOcean             3000009
DxCustomAOV             3000010
DxFoam                  3000020
DxGeometryAOVs          3000021
DxTo                    3000022
DxGrade                 3000030
DxManifold2D            3000031
DxHoudiniOcean          3000032
DxTexFile               3000050

DxBeyondScreen          3000110
DxFisheye               3000120
DxOmniDirectionalStereo 3000130
```
# ZELOS
### ZelosForTest
```
ZNeighborPointsSearchTest            0x71001
ZMeshConvertTest                     0x71002
ZMeshesConvertTest                   0x71003
ZTriangleTest                        0x71004
ZVtxLines                            0x71005
ZAABBTriangleIntersectionTest        0x71006
ZAABBLineIntersectionTest            0x71007
ZMeshToHeightFieldTest               0x71008
ZMeshTo2DLevelSetFieldTest           0x71009
ZNoiseTest                           0x71010
ZCurlNoiseTest                       0x71011
ZSampingTest                         0x71012
ZTangentSpaceTest                    0x71013
ZMeshTo3DLevelSetFieldTest           0x71014
ZCurveTest                           0x71015
ZJiggleTest                          0x71016
ZFootPrint                           0x71017
ZAddPoint                            0x71018
ZInNOutLocator                       0x71019
ZMeshSampling                        0x71020
ZCtxScope                            0x71021
ZChainFieldTest                      0x71022
ZInMeshOutMesh                       0x71023
ZTriMeshVolumeCenterOfMassTest       0x71024
ZCurveSmoothTest                     0x71025
ZTessellationTest                    0x71026
ZFftTest                             0x71027
ZShape                               0x71028
ZInNOutNode                          0x71029
ZAttrTestNode                        0x71030
ZFootPrint                           0x71031
ZPlaneTest                           0x71032
ZFrustumTest                         0x71033
```
### ZelosWater
```
ZOceanWaveDeformer                   0x83006
ZTwoOceanWaveDeformer                0x83007
ZNoiseViewer                         0x83008
ZOceanWaveViewer                     0x83009
ZKelvinWaveDeformer                  0x83012
ZWaterPtcViewer                      0x83013
ZWaterMeshViewer                     0x83014
ZFieldViewer                         0x83015
ZBreakingWaveController              0x83016
ZBreakingWaveDeformer                0x83017
ZField3DViewer                       0x83018
ZOceanPtcEmitter                     0x83019
ZTwoOceanPtcEmitter                  0x83020
ZMayaFieldViewer                     0x83021
ZGerstnerOceanDeformer               0x83022
ZCuOceanDeformer                     0x83023
ZCuOceanViewer                       0x83024
ZRippleDeformer                      0x83025
```
### ZelosCloth
```
ZelosClothCurve                      0x84001
ZelosClothCurveMerge                 0x84002
ZelosClothPatch                      0x84003
ZelosClothSeam                       0x84004
ZelosClothMesh                       0x84005
ZelosClothSimulator                  0x84006
ZelosClothCollider                   0x84007
ZelosClothPin                        0x84008
ZelosClothAttach                     0x84009
ZelosClothSpring                     0x84010
ZelosClothGlobalMatrix               0x84011
ZelosClothGlobal                     0x84012
ZelosClothGoal                       0x84013
ZelosClothWeld                       0x84014
ZelosClothAngle                      0x84015
ZelosClothCacheViewer                0x84016
```
### ZelosFur
```
ZelosFurSampler                      0x85001
ZelosFurSimulator                    0x85002
ZelosFurGenerator                    0x85003
ZelosFurCollider                     0x85004
ZelosFurGlobalMatrix                 0x85005
ZelosFurGlobal                       0x85006
ZelosFurModifier_Clump               0x85007
ZelosFurModifier_Scale               0x85008
ZelosFurModifier_Bend                0x85009
ZelosFurModifier_Direction           0x85010
ZelosFurModifier_Offset              0x85011
ZelosFurModifier_Cut                 0x85012
ZelosFurModifier_Scraggle            0x85013
ZelosFurModifier_Width               0x85014
ZelosFurModifier_Opacity             0x85015
ZelosFurTexture                      0x85020
ZelosFurTextureViewer                0x85021
ZelosFurOutput                       0x85022
ZelosFurCacheViewer                  0x85023
ZelosFurPtcBinder                    0x85024
```
### ZelosMesh
```
ZOffsetMeshDeformer                  0x90012
ZRelaxationMeshDeformer              0x90014
ZJiggleMeshDeformer                  0x82001
ZDeformationTransfer                 0x82002
ZWrapMeshDeformer                    0x82005
ZDeltaMush                           0x82008
ZAbcViewer                           0x82009
ZAbcViewerInstance                   0x82010
ZAbcViewerData                       0x82011
```
### Zeom
```
ZMeshClipper                         0x90001
ZMeshXFormBinder                     0x90002
ZMeshLoader                          0x90003
ZEnvSphere                           0x90004
ZCollisionMapGen                     0x90005
ZTreeSkeletonGen                     0x90006
```
### ZelosCamera
```
ZCameraViewFrustum                   0x91001
ZLinesFromCamera                     0x91002
```
### gpuCache
```
ZelosGpuCacheMaya                    0x92000
```
### ZelosPythonForMaya
```
ZNumToString                         0x93001

ZDarkRideGauge                       0x93003
ZGaugeMPxData                        0x93004
ZDarkRideAimLimit                    0x93005
```
### ZENN
```
ZN_GlobalMatrix                      0x100001
ZN_Global                            0x100002
ZN_GroupMatrix                       0x100003
ZN_Group                             0x100004
ZN_StrandsData                       0x100005
ZN_Import                            0x100006
ZN_MeshViewer                        0x100007
ZN_StrandsViewer                     0x100008
ZN_Generate                          0x100009
ZN_Animate                           0x100010
ZN_Select                            0x100011
ZN_Merge                             0x100012
ZN_Pass                              0x100013
ZN_Copy                              0x100014
ZN_Switch                            0x100015
ZN_Image                             0x100016
ZN_ImageData                         0x100017
ZN_DeformerData                      0x100018
ZN_Deform                            0x100019
ZN_Deform_Scale                      0x100020
ZN_Deform_Offset                     0x100021
ZN_Deform_Cut                        0x100022
ZN_Deform_Clump                      0x100023
ZN_Deform_Bend                       0x100024
ZN_Deform_Direction                  0x100025
ZN_Deform_Frizz                      0x100026
ZN_Deform_Kink                       0x100027
ZN_Save                              0x100028 // obsolete
ZN_Load                              0x100029
ZN_Deform_Width                      0x100030
ZN_Deform_Opacity                    0x100031
ZN_FeatherData                       0x100032
ZN_FeatherImport                     0x100033
ZN_FeatherSetData                    0x100034
ZN_FeatherInstance                   0x100035
ZN_FeatherSetViewer                  0x100036
ZN_FeatherDesigner                   0x100037
ZN_OutputMatrix                      0x100038
ZN_Output                            0x100039
ZN_Jiggle                            0x100040
ZN_PartialMeshGen                    0x100041
ZN_GroomMatrix                       0x100042
ZN_Groom                             0x100043
ZN_GroomBrushManip                   0x100044
ZN_GroomBrushLoc                     0x100045
ZN_CurveToMeshBinder                 0x100050
ZN_MayaCurvesGen                     0x100051
ZN_TipLocatorsGen                    0x100052
ZN_Flutter                           0x100053
ZN_Deform_Axis                       0x100054
ZN_Instance                          0x100055
ZN_FeatherMeshesGen?                 0x100056
ZN_Source                            0x100057
ZN_SourceData                        0x100058
ZN_Archive                           0x100059
ZN_StrandsArchive                    0x100060
ZN_Cut                               0x100061
ZN_Select                            0x100062
```
### ZENV
```
ZEnvPointData                        0x101001
ZEnvPointSetData                     0x101002
ZEnvSourceData                       0x101003
ZEnvSourceSetData                    0x101004
ZEnvGroupMatrix                      0x101005
ZEnvGroup                            0x101006
ZEnvPointGroupMatrix                 0x101007
ZEnvPointGroup                       0x101008
ZEnvSourceGroupMatrix                0x101009
ZEnvSourceGroup                      0x101010
ZEnvPointNode                        0x101011
ZEnvPointSetNode                     0x101012
ZEnvSourceNode                       0x101013
ZEnvSourceSetNode                    0x101014
ZEnvInstanceNode                     0x101015
ZEnvPointSeparator                   0x101016
ZEnvArchive                          0x101017
ZEnvArchiveV2                        0x101018
```
### ZAlembicForMaya
```
ZA_GpuCacheViewer                    0x110001
```
### ZelosFluid
```
ZParticlesLoader                     0x96001
```
### ZFX
```
ZFX_GridData                         0x120001
ZFX_GRID                             0x120002
ZFX_OceanData                        0x120003
ZFX_OCEAN                            0x120004
ZFX_FLIP                             0x120005
ZFX_PtcData                          0x120006
ZFX_PtcSetData                       0x120007
ZFX_PTC                              0x120008
ZFX_VIEW                             0x120009
ZFX_BreakingWave                     0x120010
ZFX_OBJECT                           0x120011
ZFX_ObjectData                       0x120012
ZFX_OceanGlobal                      0x120013
ZFX_ImageData                        0x120014
ZFX_IMAGE                            0x120015
ZFX_OceanToImage                     0x120016
ZFX_OceanGlobalData                  0x120017
ZFX_OceanViewer                      0x120018
ZFX_BreakingWaveData                 0x120019
ZFX_WaveToImage                      0x120020
ZFX_ForceCarrier                     0x120021
ZFX_PtcDSO                           0x120022
ZFX_PartialMeshGen                   0x120023
ZFX_LSF                              0x120024
ZFX_ScalarFieldViewer                0x120025
ZFX_FieldData                        0x120026
ZFX_TurbulenceField                  0x120027
ZFX_DisplaceMesh                     0x120028
ZFX_Field3DToImage                   0x120029
ZFX_VectorFieldViewer                0x120030
ZFX_PtcAdvection                     0x120031
ZFX_PtcEmitterMap                    0x120032
ZFX_VecMapOnMeshViewer               0x120033
ZFX_PointArrayData                   0x120034
ZFX_CurveBasedInstance               0x120035
ZFX_PointArrayViewer                 0x120036
ZFX_PointArrayToVolume               0x120037
ZFX_VDBGridData                      0x120038
ZFX_VDBGridViewer                    0x120039
ZFX_CurveToVolume                    0x120042
ZFX_Noise                            0x120043
ZFX_NoiseData                        0x120044
ZFX_VolumePrimData                   0x120045
ZFX_PointsToVolume                   0x120046
ZFX_VolumeViewer                     0x120047
```
### ZarVis
```
ZVNode                               0x130000
ZVPtcData                            0x130001
ZVPtcViewer                          0x130002
ZVImporter                           0x130003
ZVBreakingWave                       0x130004
ZVOceanViewer                        0x130005
ZVParticleDiffuser                   0x130006
ZVMeshData                           0x130007
ZVMeshConverter                      0x130008
ZVOceanWave                          0x130009
ZVOceanData                          0x130010
ZVOceanBlend                         0x130011
ZVFieldData                          0x130012
ZVRasterizer                         0x130013
ZVFieldViewer                        0x130014
ZVRManVolArchive                     0x130015
ZVRManPtcArchive                     0x130016
ZVRManMeshArchive                    0x130017
ZVImageData                          0x130018
ZVExporter                           0x130019
ZVImage                              0x130020
ZVLauncher                           0x130021
ZVCacheViewer                        0x130022
ZVGrid                               0x130023
ZVGridData                           0x130024
ZVVolumeDiffuser                     0x130025
ZVRManMeshArchive                    0x130026
ZVVolumeViewer                       0x130027
ZWaterTankCreator                    0x130028
```
### ZFx
```
ZX_Grid                              0x97001
ZX_ScalarField                       0x97002
ZX_VectorField                       0x97003
ZX_Ptc                               0x97004
ZX_TriMesh                           0x97005
ZX_Create_Grid                       0x97011
ZX_Create_ScalarField                0x97012
ZX_Create_VectorField                0x97013
ZX_Create_Particles                  0x97014
ZX_Create_TriMesh                    0x97015
ZX_Pass_Grid                         0x97021
ZX_Pass_ScalarField                  0x97022
ZX_Pass_VectorField                  0x97023
ZX_MeshToLevelSet                    0x97031
ZX_MeshToVelField                    0x97032
(x)ZX_MeshToParticle                 0x97033
ZX_Viewer_Field                      0x97034
ZX_Viewer_Particles                  0x97035
ZX_Viewer_Mesh                       0x97036
ZX_Advect_ScalarField                0x97037
ZX_Advect_Paritcles                  0x97038
ZX_BoundaryCondition                 0x97039
ZX_ExternalForce                     0x97040
eaX_ParticleToGrid                   0x97041
ZX_GridToParticle                    0x97042
ZX_Advect_Test                       0x97043
ZX_OceanWave                         0x97044
ZX_Viewer_TriMesh                    0x97045
ZX_OceanPreview                      0x97046
ZX_ProjectionSolver                  0x97047
ZX_AddCollider                       0x97048
ZX_AddSource                         0x97049
ZX_Advect_VectorField                0x97050
ZX_LevelsetReDistancer               0x97051
ZX_LevelsetToMesh                    0x97052
```
### ZRigForMaya
```
ZCurveDivider                        0x98001
ZAxisVisualizer                      0x98002
```
### ZWeb
```
ZWebCurves                           0x140001
ZStitchCurves                        0x140002
ZWebCurvesControl                    0x140003
ZWebCurvesData                       0x140004
ZWStitchCurveControl                 0x140005
```
### ZMayaTools
```
ZGpuMeshData                         0x99001
ZGpuMeshCreator                      0x99002
ZGpuMeshShape                        0x99003
ZGpuMeshTrigger                      0x99004
ZAssemblyArchive                     0x99005
ZAbcPtcViewer                        0x99006
ZPtcEditor                           0x99007
ZDarkRideAim                         0x99008
ZAssemblyArchiveData                 0x99009
```
### Arachne
```
ZArachneCharacter                    0x150001
```
### BeyondScreen
```
BeyondScreenTest                     0x71034
BeyondScreenCacheViewer              0x71035
```
### BORA
```
BoraNodeData                         0x200001
BoraBegeoImporter                    0x200002
BoraVolumeFromPoints                 0x200003
BoraVDBFromPointsCmd                 0x200004

BoraNodeTemplate                     0x300000
BoraGrid                             0x300001
Bora_UniformScalarField              0x300002
BoraOcean                            0x300003
Bora_ConvertToLevelSet_FSM           0x300004
Bora_ConvertToLevelSet_FMM           0x300005
BoraTest                             0x300006
Bora_MarchingCubesTest               0x300007
BoraBreakingWave                     0x300008
BoraOceanMesh                        0x300009
BoraHeightMerge                      0x300010
BoraHMeshMerge                       0x300011
```
### TANE
```
///----------------------------------------------- MPxData
TN_POINTS_MPXDATA_ID                                    0x40000000
TN_POINTS_MPXDATA_NAME                                  "TN_PointsMPxData"
TN_SOURCE_MPXDATA_ID                                    0x40000001
TN_SOURCE_MPXDATA_NAME                                  "TN_SourceMPxData"
TN_SOURCES_MPXDATA_ID                                   0x40000003
TN_SOURCES_MPXDATA_NAME                                 "TN_SourcesMPxData"
TN_DIRTYFLAG_MPXDATA_ID                                 0x40000004
TN_DIRTYFLAG_MPXDATA_NAME                               "TN_DirtyFlagMPxData"
TN_MESSAGE_MPXDATA_ID                                    0x40000005
TN_MESSAGE_MPXDATA_NAME                                  "TN_MessageMPxData"
///----------------------------------------------- MPxNode
//TN_DISTRIBUTOR_MPXNODE_ID                             0x41000000
//TN_DISTRIBUTOR_MPXNODE_NAME                           "TN_EnvironmentMPxNode"
//TN_DISTRIBUTOR_MPXNODE_PRETTY_NAME                    "TN_Environment"
TN_ENVIRONMENT_MPXNODE_ID                               0x41000000
TN_ENVIRONMENT_MPXNODE_NAME                             "TN_EnvironmentMPxNode"
TN_ENVIRONMENT_MPXNODE_PRETTY_NAME                      "TN_Environment"
TN_CONVERTZENV_MPXNODE_ID                               0x41000006
TN_CONVERTZENV_MPXNODE_NAME                             "TN_ConvertZEnvMPxNode"
TN_CONVERTZENV_MPXNODE_PRETTY_NAME                      "TN_ConvertZEnv"
TN_IMPORTCACHE_MPXNODE_ID                               0x41000007
TN_IMPORTCACHE_MPXNODE_NAME                             "TN_ImportCacheMPxNode"
TN_IMPORTCACHE_MPXNODE_PRETTY_NAME                      "TN_ImportCache"
///----------------------------------------------- MPxTransformationMatrix
///----------------------------------------------- MPxTransform
TN_TANE_MPXTRANSFORM_MATRIX_ID                          0x42000000
TN_TANE_MPXTRANSFORM_ID                                 0x42000001
TN_TANE_MPXTRANSFORM_NAME                               "TN_TaneMPxTransform"
TN_TANE_MPXTRANSFORM_PRETTY_NAME                        "TN_Tane"
TN_ARCHIVE_MPXTRANSFORM_MATRIX_ID                       0x42000002
TN_ARCHIVE_MPXTRANSFORM_ID                              0x42000003
TN_ARCHIVE_MPXTRANSFORM_NAME                            "TN_ArchiveMPxTransform"
TN_ARCHIVE_MPXTRANSFORM_PRETTY_NAME                     "TN_Archive"
///----------------------------------------------- MPxSurfaceShape
TN_SOURCEIMPORT_MPXSURFACESHAPE_ID                      0x43000000
TN_SOURCEIMPORT_MPXSURFACESHAPE_NAME                    "TN_SourceImportMPxSurfaceShape"
TN_SOURCEIMPORT_MPXSURFACESHAPE_PRETTY_NAME             "TN_SourceImportShape"
TN_TANE_MPXSURFACESHAPE_ID                              0x43000001
TN_TANE_MPXSURFACESHAPE_NAME                            "TN_TaneMPxSurfaceShape"
TN_TANE_MPXSURFACESHAPE_PRETTY_NAME                     "TN_TaneShape"
TN_ABCPROXY_MPXSURFACESHAPE_ID                          0x43000002
TN_ABCPROXY_MPXSURFACESHAPE_NAME                        "TN_AbcProxyMPxSurfaceShape"
TN_ABCPROXY_MPXSURFACESHAPE_PRETTY_NAME                 "TN_AbcProxyShape"
TN_ARCHIVE_MPXSURFACESHAPE_ID                           0x43000003
TN_ARCHIVE_MPXSURFACESHAPE_NAME                         "TN_ArchiveMPxSurfaceShape"
TN_ARCHIVE_MPXSURFACESHAPE_PRETTY_NAME                  "TN_ArchiveShape"
///----------------------------------------------- MPxManipContainer
TN_TANE_MPXSURFACESHAPE_MANIP_ID                        0x44000000
//TN_TANE_MPXSURFACESHAPE_MANIP_NAME                    "TN_TaneMPxSurfaceShape"
```
### phenoface
```
ARAPDeformer 0x71041
ARAPLocator  0x71042
ARAPData     0x71043
```
