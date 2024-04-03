//-----------//
// ZLogger.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZLogger_h_
#define _ZLogger_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// ex) ZLogger log; log.dump( true, false, "%s\n", "abc" );

class ZLogger
{
	protected:

		ZString _filePathName;

	public:

		ZLogger();

		void setFilePathName( const char* filePathName );

		ZString getFilePathName() const;

		void deleteFile() const;

		void dump( const char* format, ... ) const;
		void dump( bool onConsole, bool onFileDump, const char* format, ... ) const;
};

ostream&
operator<<( ostream& os, const ZLogger& object );

ZELOS_NAMESPACE_END

#endif

