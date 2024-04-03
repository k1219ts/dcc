//-------------------//
// ZAlembicArchive.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.02.15                               //
//-------------------------------------------------------//

#ifndef _ZAlembicArchive_h_
#define _ZAlembicArchive_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A class wrapping Alembic::Abc::IArchive.
class ZAlembicArchive
{
	private:

		Alembic::Abc::IArchive         _archive;
		Alembic::Abc::ArchiveReaderPtr _archivePtr; ///< a pointer to the archive

		bool             _opened;
		int              _archiveType;		///< The archive type (0:none, 1:HDF5, 2:Ogawa)
		ZString          _filePathName;		///< The file path and name
		ZAlembicObject   _topObj;			///< The top object of the tree

		int              _abcVersion;		///< Alembic version
		ZString          _abcVersionStr;	///< Alembic version as a string
		ZString          _appName;			///< Application Name
		ZString          _writtenDate;		///< written date
		ZString          _description;		///< user description


		// time sampling info.
		int              _timeSamplingType;	///< the time sampling
		int              _numTimeSamples;	///< the number of time sampling
		double           _startTime;		///< the start time
		double           _endTime;			///< the end  time
		double           _timeStepSize;		///< the average time step size

	public:

		ZAlembicArchive();

		ZAlembicArchive( const char* filePathName );

		~ZAlembicArchive();

		void reset();

		Alembic::Abc::IArchive& archive();

		const Alembic::Abc::IArchive& archive() const;

		bool open( const char* filePathName );

		bool opened() const;

		void close();

		bool getMetaData( ZStringArray& keys, ZStringArray& values ) const;

		const ZAlembicObject& topObject() const;

		ZString archiveTypeStr() const;

		int archiveType() const;

		ZString filePathName() const;

		int abcVersion() const;

		ZString abcVersionStr() const;

		ZString appName() const;

		ZString writtenDate() const;

		ZString description() const;

		int timeSamplingTypeId() const;

		ZString timeSamplingTypeStr() const;

		int numTimeSamples() const;

		double startTime() const;

		double endTime() const;

		double timeStepSize() const;
};

ostream&
operator<<( ostream& os, const ZAlembicArchive& object );

ZELOS_NAMESPACE_END

#endif

