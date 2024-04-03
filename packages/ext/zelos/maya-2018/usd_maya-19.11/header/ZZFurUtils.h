//--------------//
// ZZFurUtils.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2015.04.10                               //
//-------------------------------------------------------//

#ifndef _ZZFurUtils_h_
#define _ZZFurUtils_h_

void
zSortFurSegments
(
	int numTotalSegments, zXYZ camPosition,
	int* sv0, zXYZ* sp0, float* dst,
	int* v01
);

#endif

