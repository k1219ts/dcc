//--------------//
// ZCudaUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
//         Jinhyuk Bae @ Dexter Studios                  //
// last update: 2014.05.28                               //
//-------------------------------------------------------//

#ifndef _ZCudaUtils_h_
#define _ZCudaUtils_h_

#include <ZelosCudaBase.h>

ZString zCudaInfo();
void ZPrintCudaInfo();

// Return the best GPU (with maximum GFLOPS).
// If failed, it will return -1.
int ZBestCudaDeviceId();

// Sets device as the current device for the calling host thread.
bool ZSetCurrentCudaDevice( int deviceId=0 );

// This call may be made from any host thread, to any device, and at any time.
// This function will do no synchronization with the previous or new device,
// and should be considered a very low overhead call.
bool ZCudaInit();

// Explicitly destroys and cleans up all resources associated with the current device in the current process.
void ZCudaExit();

#endif

