//---------//
// ZAxis.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.01                               //
//-------------------------------------------------------//

#ifndef _ZAxis_h_
#define _ZAxis_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZAxis
{
	public:

		ZPoint origin;		///< The origin.

		ZVector xAxis;		///< The x-axis.
		ZVector yAxis;		///< The y-axis.
		ZVector zAxis;		///< The z-axis.

	public:

		/// @brief The default constructor.
		/**
			Create a world axis-aligned unit axis.
		*/
		ZAxis();

		/// @brief The copy constructor.
		/**
			Create a new axis and set the axis as a world axis-aligned unit one.
		*/
		ZAxis( const ZAxis& source );

		/// @brief The re-initializer.
		/**
			Set the axis as a world axis-aligned unit one.
 		*/
		void reset();

		/// @brief The assignement operator.
		/**
			Copy all of the data of the other axis instance into this one.
		*/
		ZAxis& operator=( const ZAxis& source );

		/// @brief The re-initializer.
		/**
			Set the axis form the given three axes and their barycentric coordinates.
		*/
		void set( const ZAxis& a0, const ZAxis& a1, const ZAxis& a2, float w0, float w1, float w2, bool robust=true );

		ZAxis& robustNormalize();
		ZAxis& normalize( bool accurate=false );

		void changeHandedness( const int& i );

		ZVector worldToLocal( const ZVector& worldPosition, bool asVector, bool hasNormalizedAxes=true ) const;
		ZVector localToWorld( const ZVector& localPosition, bool asVector, bool hasNormalizedAxes=true ) const;

		bool align( const ZVector& aim, ZDirection::Direction whichAxis, bool accurate );

		void draw( bool bySimpleLine=true ) const;
};

ostream&
operator<<( ostream& os, const ZAxis& object );

ZELOS_NAMESPACE_END

#endif

