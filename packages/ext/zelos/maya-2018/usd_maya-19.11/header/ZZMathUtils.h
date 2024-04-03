//---------------//
// ZZMathUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.15                               //
//-------------------------------------------------------//

#ifndef _ZZMathUtils_h_
#define _ZZMathUtils_h_

#include <ZelosCudaBase.h>

#define ZZAbs(x)			(((x)>0)?(x):(-x))
#define ZZPow2(x)			((x)*(x))
#define ZZPow3(x)			((x)*(x)*(x))
#define ZZClamp(x,min,max)	(((x)<(min))?((min)):(((x)>(max))?((max)):(x)))
#define ZZMin(x,y)			(((x)<(y))?(x):(y))
#define ZZMax(x,y)			(((x)>(y))?(x):(y))
#define ZZLerp(a,b,t)		(((t)<0)?(a):(((t)>1)?(b):((1-(t))*(a)+(t)*(b))))

#define ZZRadToDeg(x)		((x)*(Z_RADtoDEG))
#define ZZDegToRad(x)		((x)*Z_DEGtoRAD)

#define ZZCeil(x)			((int)(x)+(((x)>0)&&((x)!=(int)(x))))
#define ZZFloor(x)			((int)(x)-(((x)<0)&&((x)!=(int)(x))))
#define ZZTrunc(x)			((int)(x))
#define ZZRound(x)			((x)>0?(int)((x)+.5):-(int)(.5-(x)))
#define ZZFrac(x)			((x)-(int)(x))
#define ZZStep(y,x)			(((x)>=(y))?1:0)
#define ZZSgn(x)			(((x)>0)?+1:-1)
#define ZZSign(x)			(((x)<0)?-1:((x)>0)?1:0)

#endif

