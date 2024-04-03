//----------//
// ZAsser.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.09.08                               //
//-------------------------------------------------------//

#ifndef _ZAsser_h_
#define _ZAsser_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////
#define ZASSERT( TEST )                                                       \
do                                                                            \
{                                                                             \
    if( !( TEST ) )                                                           \
    {                                                                         \
        std::stringstream ss;                                                 \
        ss << endl;                                                           \
        ss << "ERROR by ZASSERT" << endl;                                     \
        ss << "  File   : " << __FILE__ << endl;                              \
        ss << "  Line   : " << __LINE__ << endl;                              \
        throw std::runtime_error( ss.str() );                                 \
    }                                                                         \
}                                                                             \
while( 0 )

///////////////////////////////////////////////////////////////////////////////
#define ZASSERT_MESSAGE( TEST, MSG )                                          \
do                                                                            \
{                                                                             \
    if( !( TEST ) )                                                           \
    {                                                                         \
        std::stringstream ss;                                                 \
        ss << endl;                                                           \
        ss << "ERROR by ZASSERT" << endl;                                     \
        ss << "  File   : " << __FILE__ << endl;                              \
        ss << "  Line   : " << __LINE__ << endl;                              \
        ss << "  Message: " << MSG      << endl;                              \
        throw std::runtime_error( ss.str() );                                 \
    }                                                                         \
}                                                                             \
while( 0 )

ZELOS_NAMESPACE_END

#endif

