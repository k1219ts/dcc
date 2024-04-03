#ifndef _BS_FileUtils_h_
#define _BS_FileUtils_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

static bool DoesFileExist( const char* filePathName )
{
    struct stat buffer;
    const int exist = stat( filePathName, &buffer );
    return ( (exist==0) ? true : false );
}

static String FileExtension( const char* filePathName )
{
    String fileStr( filePathName );

    StringArray tokens;
    tokens.setByTokenizing( fileStr, "." );

    return tokens.last();
}

static bool IsDirectory( const char* path )
{
    struct stat fstat;
    lstat( path, &fstat );
    if( S_ISDIR(fstat.st_mode) ) { return true; }
    return false;
}

static bool GetFileList( const char* path, StringArray& files, bool asFullPath=false )
{
    files.clear();

    DIR* dp;
    struct dirent* dirp;

    if( !( dp = opendir( path ) ) )
    {
        cout << "Error@GetFileList(): Failed open file " << path << endl;
        return false;
    }

    while( ( dirp = readdir(dp) ) != NULL )
    {
        const std::string shortPath = dirp->d_name;

        std::string fullPath = path;
        fullPath += "/" + shortPath;

        if( IsDirectory( fullPath.c_str() ) ) { continue; }

        if( asFullPath )
        {
            files.push_back( fullPath );
        }
        else
        {
            files.push_back( shortPath );
        }
    }

    closedir( dp );

	return true;
}

static bool GetFileList( const char* path, const std::string& extension, StringArray& files, bool asFullPath=false )
{
    StringArray candidates;

    if( !GetFileList( path, candidates, asFullPath ) )
    {
        return false;
    }

	const size_t numFiles = candidates.size();

    for( size_t i=0; i<numFiles; ++i )
	{
        const std::string ext = FileExtension( candidates[i].c_str() );

		if( ext == extension )
		{
			files.push_back( candidates[i] );
		}
	}

	return true;
}

static bool CreateDirectory( const char* path, mode_t permission=0755 )
{
	if( DoesFileExist( path ) ) { return true; }

	String tmp( path );

	StringArray tokens;
    tokens.setByTokenizing( tmp, "/" );

	const int N = tokens.length();
	if( N == 0 ) { return false; }
	tmp.clear();
	
	for( int i=0; i<N; ++i )
	{
		tmp += "/" + tokens[i];
		if( !DoesFileExist( tmp.c_str() ) )
		if( mkdir( tmp.c_str(), permission ) )
		{
			cout << "Error@CreateDirectory(): Failed to create a directory." << endl;
			return false;
		}
	}

	return true;
}

BS_NAMESPACE_END

#endif

