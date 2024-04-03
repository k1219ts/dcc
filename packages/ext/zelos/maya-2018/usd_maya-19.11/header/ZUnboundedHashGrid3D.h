//------------------------//
// ZUnboundedHashGrid3D.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.28                               //
//-------------------------------------------------------//

/// @brief Hash grid class

#ifndef _ZUnboundedHashGrid3D_h_
#define _ZUnboundedHashGrid3D_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A class for an unbounded spatial partitioning hash grid.
class ZUnboundedHashGrid3D
{
	private:

		#ifdef HIGH_GCC_VER

			// cell index (ZInt3) -> hash key (int)
			struct Func : public std::unary_function<ZInt3,int>
			{
				int operator()( const ZInt3& i ) const
				{
					return ( i[0]*0x8da6b343 ^ i[1]*0xd8163841 ^ i[2]*0xcb1ab31f );
				}
			};
			typedef typename tr1::unordered_multimap<ZInt3,int,Func> Hash;

		#else

			struct Func
			{
				bool operator()( const ZInt3& lhs, const ZInt3& rhs ) const
				{
					const int i0 = lhs[0]*0x8da6b343 ^ lhs[1]*0xd8163841 ^ lhs[2]*0xcb1ab31f;
					const int i1 = rhs[0]*0x8da6b343 ^ rhs[1]*0xd8163841 ^ rhs[2]*0xcb1ab31f;
					return ( i0 < i1 );
				}
			};
			typedef typename std::multimap<ZInt3,int,Func> Hash;

		#endif

		typedef Hash::iterator       HashItr;
		typedef Hash::const_iterator HashConstItr;

	private:

		float _h;		// the cell size
		float _hInv;	// the inverse of the cell size

		Hash  _hash;

	public:

		ZUnboundedHashGrid3D();
		ZUnboundedHashGrid3D( float h );

		void reset();
		
		void set( float h );

		bool empty() const;
		bool empty( const ZInt3& cell ) const;

		int numElements( const ZInt3& cell ) const;

		void getAllocatedCells( ZInt3Array& cells ) const;

		void addItem( int id, const ZPoint& p );
		void addItem( int id, const ZBoundingBox& bBox );

		int firstItem( const ZInt3& cell ) const;

		// Caution)
		// candidates[i] may not be inside the given sphere.
		// It's only broad-band test!
		void getCandidates( ZIntArray& candidates, const ZPoint& p, float radius, bool removeDuplications=false, bool asAppending=false ) const;

		void findPoints( const ZPoint& p, float radius, ZIntArray& pointIds, ZFloatArray& distSQ, const ZPointArray& sP ) const;

		void removeAllItems( const ZInt3& cell );
		void remove( int id );

		// for the Poisson disk sampling on a surface
		void remove( const ZPoint& p, float raiuds, const ZPointArray& sP );
		void remove( const ZPoint& p, const ZVector& n, float raiuds, const ZPointArray& sP, const ZVectorArray& sN );

		float cellSize() const;
		ZPoint cellCenter( const ZInt3& cell ) const;
};

ostream&
operator<<( ostream& os, const ZUnboundedHashGrid3D& object );

ZELOS_NAMESPACE_END

#endif

