//-*****************************************************************************
//
// Copyright (c) 2009-2011,
//  Sony Pictures Imageworks Inc. and
//  Industrial Light & Magic, a division of Lucasfilm Entertainment Company Ltd.
//
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// *       Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
// *       Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
// *       Neither the name of Sony Pictures Imageworks, nor
// Industrial Light & Magic, nor the names of their contributors may be used
// to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//-*****************************************************************************
#include "SampleUtil.h"
#include <ri.h>

//-*****************************************************************************
void WriteMotionBegin( ProcArgs &args, const SampleTimeSet &sampleTimes )
{
    std::vector<RtFloat> outputTimes;
    outputTimes.reserve( 2 );

    if( args.subframe == 1 )
    {
        outputTimes.push_back( 0 );
        if( args.shutterOpen < 0.0 ) {
            if( args.shutterClose > 0.0 )
                outputTimes.push_back( args.shutterClose*2 );
            else
                outputTimes.push_back( args.shutterOpen*-1 );
        } else {
            outputTimes.push_back( args.shutterClose );
        }
    }
    else
    {
        outputTimes.push_back( 0 );
        outputTimes.push_back( 1 );
    }

    RiMotionBeginV( outputTimes.size(), &outputTimes[0] );
}

//-*****************************************************************************
void WriteConcatTransform( const M44d &m )
{
    RtMatrix rtm;

    for ( int row = 0; row < 4; ++row )
    {
        for ( int col = 0; col < 4; ++col )
        {
            rtm[row][col] = (RtFloat)( m[row][col] );
        }
    }

    RiConcatTransform( rtm );
}


chrono_t LoopFrame( chrono_t startFrame, chrono_t endFrame, double inFrame )
{
    chrono_t in_start = inFrame - startFrame;
    chrono_t duration = endFrame - startFrame + 1;
    chrono_t frame = in_start - ( std::floor(in_start/duration) * duration ) + startFrame;
    return frame;
}

chrono_t OscillateFrame( chrono_t startFrame, chrono_t endFrame, double inFrame )
{
    chrono_t frame;
    double duration = endFrame - startFrame;
    double pos      = fmod( inFrame-startFrame, duration*2 );
    if( pos >= duration )
        frame = endFrame - ( pos - duration );
    else
        frame = fmod( inFrame-startFrame, duration+startFrame );
    return frame;
}

void GetRelevantSampleTimes( ProcArgs &args, TimeSamplingPtr timeSampling,
                             size_t numSamples, SampleTimeSet &output )
{
    if( numSamples < 2 ) {
        output.insert( 0.0 );
        return;
    }

    chrono_t frame;
    chrono_t nframe;
    if( args.cycle > 0 )
    {
        chrono_t startFrame = timeSampling->getSampleTime( 0 ) * args.fps;
        chrono_t endFrame   = timeSampling->getSampleTime( numSamples-1 ) * args.fps;
        double shutterOpen  = args.shutterOpen;
        double shutterClose = args.shutterClose;
        // loop
        if( args.cycle == 1 ) {
            frame  = LoopFrame( startFrame, endFrame, args.frame+shutterOpen );
            nframe = LoopFrame( startFrame, endFrame, args.frame+shutterClose );
        } else {
        // oscillate
            frame  = OscillateFrame( startFrame, endFrame, args.frame+shutterOpen );
            nframe = OscillateFrame( startFrame, endFrame, args.frame+shutterClose );
        }
    }
    else
    {
        frame  = args.frame + args.shutterOpen;
        nframe = args.frame + args.shutterClose;
    }

    chrono_t shutterOpenTime  = frame / args.fps;
    chrono_t shutterCloseTime = nframe / args.fps;

//    std::cout << "frame : " << frame << ", " << nframe << std::endl;

    output.insert( shutterOpenTime );
    output.insert( shutterCloseTime );

    if( output.size() == 0 ) {
        chrono_t frameTime = frame / args.fps;
        output.insert( frameTime );
    }

//    std::cout << "size : " << output.size() << std::endl;
}
