//-------------//
// ZelosBase.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.04.19                               //
//-------------------------------------------------------//

/// @mainpage Zelos Help
/// @section a Introduction
/// - Zelos is a C++ based in-house CG toolkit for Dexter Studios.
/// - It includes lots of classes and functions about mathematics, geometry, physics, etc.
/// - Zelos is to be used for fluid, hair/fur, shape deformation, rigid body, etc.
/// - It is released as various types such as dynamic/static library, Maya plug-in, Python module, etc.
/// - It will be upgraded gradually by adding new features by Dexter Studios R&D team.
/// @section b Developer
/// - Wanho Choi @ Dexter Studios
/// - Jinhyuk Bae @ Dexter Studios
/// - Nayoung Kim @ Dexter Studios
/// - Jaegwang Lim @ Dexter Studios
/// - Julie Jang @ Dexter Studios
/// - Dohyun Yang @ Dexter Studios
/// - Jungmin Lee @ Dexter Studios

#ifndef _ZelosBase_h_
#define _ZelosBase_h_

#define ZELOS_VERSION 1.0

// OS selection
#define OS_LINUX
//#define OS_WINDOWS

//=========================================================================================================//
// Base                                                                                                    // 
//=========================================================================================================//

#if( (__GNUC__==4) && (__GNUC_MINOR__>7) )
#define HIGH_GCC_VER
#endif

#include <cmath>
#include <ctime>
#include <bitset>
#include <cstdio>
#include <cfloat>
#include <cctype>
#include <cstdarg>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <typeinfo>
#include <fcntl.h>

#include <map>
#include <set>
#include <list>
#include <stack>
#include <queue>
#include <vector>
#include <iterator>
#include <algorithm>

#include <thread>

#ifdef HIGH_GCC_VER
 #include <tr1/unordered_map>
#endif

#include <omp.h>
#include <zlib.h>

#ifdef OS_WINDOWS
 #include <windows.h>
 #include <direct.h>
 #include <io.h>
 #pragma comment ( lib, "ws2_32.lib" ) // for 'gethostname()'
 #pragma comment ( lib, "glut32.lib" )
#endif

#ifdef OS_LINUX
 #include <errno.h>
 #include <netdb.h>       // for 'getnameinfo()'
 #include <climits>
 #include <dirent.h>      // DIR structure
 #include <stdint.h>
 #include <ifaddrs.h>     // for 'getifaddrs()'
 #include <sys/stat.h>
 #include <sys/time.h>
 #include <sys/utsname.h>
 #include <sys/utsname.h>
#endif

#if defined( __linux__ )
	#include <tbb/concurrent_vector.h>
	#define CONCURRENCY tbb
#else
	#include <concurrent_vector.h>
	#define CONCURRENCY concurrency
#endif

using namespace std;

//=========================================================================================================//
// OpenGL                                                                                                  // 
//=========================================================================================================//
#include <GL/glew.h>
#include <GL/glut.h>

//=========================================================================================================//
// Zelos                                                                                                   // 
//=========================================================================================================//

#include <ZFoundation.h>
#include <ZAssert.h>
#include <ZDebugLog.h>
#include <ZTestClass.h>
#include <ZSTLUtils.h>
#include <ZString.h>
#include <ZMemoryUtils.h>
#include <ZMathUtils.h>
#include <ZRandom.h>
#include <ZLogger.h>
#include <ZTimer.h>

#include <ZDataUnit.h>
#include <ZDataType.h>
#include <ZFMMState.h>
#include <ZDirection.h>
#include <ZColorSpace.h>
#include <ZImageFormat.h>
#include <ZRotationOrder.h>
#include <ZCurvatureType.h>
#include <ZFieldLocation.h>
#include <ZSamplingMethod.h>
#include <ZComputingMethod.h>
#include <ZMeshElementType.h>
#include <ZMeshDisplayMode.h>
#include <ZPointDisplayMode.h>

#include <ZTuple.h>
#include <ZVector.h>
#include <ZColor.h>
#include <ZMatrix.h>
#include <ZComplex.h>
#include <ZQuaternion.h>

#include <ZEquationSolvers.h>

#include <ZAxis.h>
#include <ZCalcUtils.h>
#include <ZTupleUtils.h>
#include <ZQuaternionUtils.h>

#include <ZBoundingBox.h>
#include <ZBox.h>

#include <ZRay.h>
#include <ZLine.h>
#include <ZPlane.h>
#include <ZTriangle.h>
#include <ZSphere.h>
#include <ZFrustum.h>

#include <ZGlUtils.h>
#include <ZTrackBall.h>
#include <ZGlCamera.h>
#include <ZGlBitmap.h>
#include <ZGlTex2D.h>
#include <ZGlTex3D.h>
#include <ZGlRbo.h>
#include <ZGlFbo.h>
#include <ZGlVbo.h>
#include <ZGlslShader.h>
#include <ZGlslProgram.h>

#include <ZHeap.h>

#include <ZList.h>
#include <ZIntList.h>
#include <ZFloatList.h>
#include <ZDoubleList.h>

#include <ZArray.h>
#include <ZCharArray.h>
#include <ZUCharArray.h>
#include <ZIntArray.h>
#include <ZInt2Array.h>
#include <ZInt3Array.h>
#include <ZInt4Array.h>
#include <ZFloatArray.h>
#include <ZFloat2Array.h>
#include <ZFloat3Array.h>
#include <ZFloat4Array.h>
#include <ZDoubleArray.h>
#include <ZDouble2Array.h>
#include <ZDouble3Array.h>
#include <ZDouble4Array.h>
#include <ZVectorArray.h>
#include <ZComplexArray.h>
#include <ZQuaternionArray.h>
#include <ZMatrixArray.h>
#include <ZAxisArray.h>
#include <ZColorArray.h>
#include <ZStringArray.h>
#include <ZBoundingBoxArray.h>

#include <ZIntArrayList.h>
#include <ZFloatArrayList.h>
#include <ZDoubleArrayList.h>
#include <ZIntSetArray.h>
#include <ZFloatSetArray.h>
#include <ZDoubleSetArray.h>
#include <ZVectorSetArray.h>

#include <ZArrayUtils.h>
#include <ZMatrixUtils.h>
#include <ZStringUtils.h>
#include <ZSystemUtils.h>
#include <ZFileUtils.h>
#include <ZJSON.h>
#include <ZMetaData.h>

#include <ZDenseMatrix.h>
#include <ZSparseMatrix.h>
#include <ZDenseMatrixUtils.h>
#include <ZSparseMatrixUtils.h>
#include <ZLinearSystemSolver.h>

#include <ZSimplexNoise.h>
#include <ZCurlNoise.h>

#include <ZSamplingUtils.h>

#include <ZImageMap.h>
#include <ZImageMapUtils.h>

#include <ZGrid2D.h>
#include <ZGrid3D.h>
#include <ZHashGrid2D.h>
#include <ZHashGrid3D.h>
#include <ZUnboundedHashGrid3D.h>

#include <ZCurve.h>
#include <ZCurves.h>

#include <ZTriMesh.h>
#include <ZTriMeshConnectionInfo.h>
#include <ZTriMeshUtils.h>
#include <ZTriMeshIO.h>
#include <ZTriMeshScatter.h>

#include <ZQuadMesh.h>
#include <ZPolyMesh.h>

#include <ZMeshElement.h>
#include <ZMeshElementArray.h>
#include <ZMesh.h>
#include <ZMesh_Generation.h>
#include <ZMeshDistTree.h>
#include <ZPointsHashGrid.h>
#include <ZPointsDistTree.h>
#include <ZTriMeshDistTree.h>

#include <ZField2DBase.h>
#include <ZField3DBase.h>
#include <ZMarkerField2D.h>
#include <ZMarkerField3D.h>
#include <ZScalarField2D.h>
#include <ZScalarField3D.h>
#include <ZVectorField2D.h>
#include <ZVectorField3D.h>
#include <ZComplexField2D.h>
#include <ZField2DUtils.h>
#include <ZField3DUtils.h>
#include <ZLevelSet2DUtils.h>
#include <ZLevelSet3DUtils.h>
#include <ZVoxelizer.h>
#include <ZGlslVolume.h>

#include <ZParticles.h>
#include <ZParticleSet.h>

#include <ZPseudoSpring.h>

#include <ZKmeanClustering.h>

#include <ZGlslOcean.h>

#include <ZPtc.h>

#include <ZDelaunay2D.h>

#include <ZSkeleton.h>

/////////////
// Alembic //
/////////////

#include <ZAlembicCommon.h>
#include <ZAlembicProperty.h>
#include <ZAlembicPropertyArray.h>
#include <ZAlembicObject.h>
#include <ZAlembicObjectArray.h>
#include <ZAlembicUtils.h>
#include <ZAlembicArchive.h>

#endif

