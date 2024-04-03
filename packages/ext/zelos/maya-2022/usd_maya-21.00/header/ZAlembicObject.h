//------------------//
// ZAlembicObject.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.04.03                               //
//-------------------------------------------------------//

#ifndef _ZAlembicObject_h_
#define _ZAlembicObject_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A class wrapping Alembic::IObject.
class ZAlembicObject
{
	private:

		Alembic::Abc::ObjectHeader _header;
		Alembic::Abc::IObject      _object;

	private:

		ZString          _name;				///< the name of the object
		ZString          _fullPath;			///< the full path of the object
		int              _typeId;			///< the object type ID
		int              _numChildren;		///< the number of children
		ZAlembicProperty _topPrp;			///< the top property of the object

		// time sampling info.
		int              _timeSamplingType;	///< the time sampling
		int              _numTimeSamples;	///< the number of time sampling
		double           _minTime;			///< the min. time
		double           _maxTime;			///< the max. time
		double           _timeStepSize;     ///< the time step size
		double           _minFrame;         ///< the min. frame
		double           _maxFrame;         ///< the max. frame

	public:

		ZAlembicObject();

		ZAlembicObject( const ZAlembicObject& obj );

		void reset();

		void set( const Alembic::Abc::IObject& obj );

		Alembic::Abc::IObject& object();

		const Alembic::Abc::IObject& object() const;

		Alembic::Abc::ObjectHeader& header();

		const Alembic::Abc::ObjectHeader& header() const;

		bool getMetaData( ZStringArray& keys, ZStringArray& values ) const;

		bool getChild( int i, ZAlembicObject& childObject ) const;

        bool getChild( const ZString& iChildName, ZAlembicObject& childObject ) const;


		bool getParent( ZAlembicObject& parentObject ) const;

		ZAlembicProperty& topProperty();

		const ZAlembicProperty& topProperty() const;

		ZAlembicObject& operator=( const ZAlembicObject& obj );

		ZString name() const;

		ZString fullPath() const;

		int typeId() const;

		ZString typeStr() const;

		bool isShape() const;

		int numChildren() const;

		bool isLeaf() const;

		int timeSamplingTypeId() const;

		ZString timeSamplingTypeStr() const;

		int numTimeSamples() const;

		bool isConstant() const;

		double minTime() const;

		double maxTime() const;

		double timeStepSize() const;

		double minFrame() const;

		double maxFrame() const;

		bool getTransformations( ZVector& t, ZVector& r, ZVector& s, int frame=0 ) const;

		bool getXFormMatrix( ZMatrix& xform, int frame = 0 ) const;

		bool getPolygonMeshInfo( int& numVertices, int& numPolygons, int& numUVs, int frame=0 ) const;

		// Caution) vConnections and uvIndices must be reversed per each polygon.
		// Use the ReverseConnections() function in ZArrayUtils.h.
		bool getPolyMeshData
		(
			ZPointArray*  vPos,
			ZVectorArray* vVel,
			ZIntArray*    vCounts,
			ZIntArray*    vConnections,
			ZBoundingBox* bBox,
			ZMatrix*      worldMat,
			ZFloatArray*  uvs,
			ZIntArray*    uvIndices,
			int frame = 0
		) const;

		int particleCount( int frame=0 ) const;
		bool getParticlePositions( ZPointArray& positions, int frame=0 ) const;
		bool getParticleIds( ZIntArray& ids, int frame=0 ) const;
		bool getParticleVelocities( ZVectorArray& velocities, int frame=0 ) const;
		bool getParticleAABB( ZBoundingBox& aabb, int frame=0 ) const;

        bool valid() const;

};

ostream&
operator<<( ostream& os, const ZAlembicObject& object );

ZELOS_NAMESPACE_END

#endif

