//-----------------//
// ZelosCudaBase.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.16                               //
//-------------------------------------------------------//

#ifndef _ZelosCudaBase_h_
#define _ZelosCudaBase_h_

#include <ZelosBase.h>
using namespace Zelos;

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h> 
#include <cuda_gl_interop.h>

#include <ZZFoundation.h>
#include <ZZMathUtils.h>
#include <ZZRandom.h>
#include <ZZFloat3Utils.h>
#include <ZZVector.h>
#include <ZZCalcUtils.h>
#include <ZZComplex.h>
#include <ZZCharArray.h>
#include <ZZIntArray.h>
#include <ZZIntSetArray.h>
#include <ZZFloatSetArray.h>
#include <ZZFloatArray.h>
#include <ZZSimplexNoise.h>
#include <ZZCurve.h>

#include <ZCudaUtils.h>
#include <ZCudaArray.h>
#include <ZCudaVbo.h>

#include <ZZFurUtils.h>

extern "C" void ZZJuliaTest( unsigned char* dev_bitmap, int dim );

#endif

