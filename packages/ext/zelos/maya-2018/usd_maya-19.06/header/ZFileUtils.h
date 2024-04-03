//--------------//
// ZFileUtils.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2018.03.19                               //
//-------------------------------------------------------//

#ifndef _ZFileUtils_h_
#define _ZFileUtils_h_

ZELOS_NAMESPACE_BEGIN

// Check whether the given path represents a directory.
bool ZIsDirectory( const char* path );

// Check whether the given path represents a symbolic link.
bool ZIsSymbolicLink( const char* path );

// Check to see if the given file exists and is readable.
bool ZFileExist( const char* path );

// Return the file size of the given file.
long int ZFileSize( const char* filePathName );

ZString ZFileSizeAsString( const char* filePathName );

// Create the directory of the given path.
bool ZCreateDirectory( const char* path, mode_t permission=0755 );

// Delete the directory of the given path.
bool ZDeleteDirectory( const char* path );

// Return the current path.
ZString ZCurrentPath();

// Return the file extension from the given file name.
ZString ZFileExtension( const char* fileName );

// Return the file name without its extension.
ZString ZRemoveExtension( const char* fileName );

// All of the '/' character of the given path will be changed to '\' and vice versa.
ZString ZChangeSeparators( const char* fileName );

// Get the file list in the given path.
bool GetFileList( const char* path, ZStringArray& files, bool asFullPath=false );

// Get the file list whose file extension is same as one in the given path.
bool GetFileList( const char* path, const ZString& extension, ZStringArray& files, bool asFullPath=false );

// Get the directory list in the given path except "." and "..".
bool GetDirectoryList( const char* path, ZStringArray& directories, bool asFullPath=false );

bool ZGetFileFrames( const ZStringArray& fileNames, ZIntArray& frames );

// Read the file and return the text presented in the file as string.
char* ZTextFromFile( const char* filePathName, int& textLength );

template <class T>
void ZWrite( ofstream& fout, const T& d, bool switchEndian=false )
{
	if( switchEndian ) {
		T tmp = d;
		ZSwitchEndian( tmp );
		fout.write( (char*)&tmp, sizeof(T) );
	} else {
		fout.write( (char*)&d, sizeof(T) );
	}
}

template <class T1, class T2>
void ZWrite( ofstream& fout, const T1& d1,const T2& d2, bool switchEndian=false )
{
	ZWrite( fout, d1, switchEndian );
	ZWrite( fout, d2, switchEndian );
}

template <class T1, class T2, class T3>
void ZWrite( ofstream& fout, const T1& d1, const T2& d2, const T3& d3, bool switchEndian=false )
{
	ZWrite( fout, d1, switchEndian );
	ZWrite( fout, d2, d3, switchEndian );
}

template <class T1, class T2, class T3, class T4>
void ZWrite( ofstream& fout, const T1& d1, const T2& d2, const T3& d3, const T4& d4, bool switchEndian=false )
{
	ZWrite( fout, d1, switchEndian );
	ZWrite( fout, d2, d3, d4, switchEndian );
}

template <class T1, class T2, class T3, class T4, class T5>
void ZWrite( ofstream& fout, const T1& d1, const T2& d2, const T3& d3, const T4& d4, const T5& d5, bool switchEndian=false )
{
	ZWrite( fout, d1, switchEndian );
	ZWrite( fout, d2, d3, d4, d5, switchEndian );
}

template <class T1, class T2, class T3, class T4, class T5, class T6>
void ZWrite( ofstream& fout, const T1& d1, const T2& d2, const T3& d3, const T4& d4, const T5& d5, const T6& d6, bool switchEndian=false )
{
	ZWrite( fout, d1, switchEndian );
	ZWrite( fout, d2, d3, d4, d5, d6, switchEndian );
}

ZELOS_NAMESPACE_END

#endif

