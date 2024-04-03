//--------------------//
// ZAlembicProperty.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.03.03                               //
//-------------------------------------------------------//

#ifndef _ZAlembicProperty_h_
#define _ZAlembicProperty_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZAlembicObject;

class ZAlembicProperty
{
	public:

		Alembic::Abc::ICompoundProperty _cmpPrp;
		Alembic::AbcCoreAbstract::PropertyHeader _header;

	private:

		ZString           _name;		///< the name of the object
		int               _typeId;		///< the property type ID
		int               _numChildren;	///< the number of children (valid only for when this is a compound type)
		int               _dataTypeId;  ///< the data type ID
		int               _extent;      ///< the extent

		ZAlembicObject*   _objectPtr;   ///< the pointer to the owner object

	public:

		ZAlembicProperty();

		ZAlembicProperty( const ZAlembicProperty& prp );

		void reset();

		void setAsTop( const ZAlembicObject& ownerObject );

		bool getMetaData( ZStringArray& keys, ZStringArray& values ) const;

		bool getChild( int i, ZAlembicProperty& childProperty ) const;

		ZAlembicObject& ownerObject();

		const ZAlembicObject& ownerObject() const;

		ZAlembicProperty& operator=( const ZAlembicProperty& prp );

		ZString name() const;

		int typeId() const;

		ZString typeStr() const;

		int numChildren() const;

		int dataTypeId() const;

		ZString dataTypeStr() const;

		int extent() const;

		//////////////////
		// simple print //

		ZString valuesAsString( int sampleIndex=0 ) const;

		////////////
		// scalar //

		// (unsigned) char
		char valueChar( int sampleIndex=0 ) const;
		bool getValue( char& value, int sampleIndex=0 ) const;

		// bool, (unsigned) short, (unsigned) int, (unsigned) long int
		int valueInt( int sampleIndex=0 ) const;
		bool getValue( int& value, int sampleIndex=0 ) const;

		// half, float
		float valueFloat( int sampleIndex=0 ) const;
		bool getValue( float& value, int sampleIndex=0 ) const;

		// double
		double valueDouble( int sampleIndex=0 ) const;
		bool getValue( double& value, int sampleIndex=0 ) const;

		// string
		ZString valueString( int sampleIndex=0 ) const;
		bool getValue( ZString& value, int sampleIndex=0 ) const;

		///////////
		// array //

		bool getValues( ZCharArray& values, int sampleIndex=0 ) const;

		bool getValues( ZIntArray& values, int sampleIndex=0 ) const;

		bool getValues( ZFloatArray& values, int sampleIndex=0 ) const;

		bool getValues( ZDoubleArray& values, int sampleIndex=0 ) const;

		bool getValues( ZStringArray& values, int sampleIndex=0 ) const;

		bool getValues( ZFloat2Array& values, int sampleIndex=0 ) const;

		bool getValues( ZFloat3Array& values, int sampleIndex=0 ) const;

		bool getValues( ZQuaternionArray& values, int sampleIndex=0 ) const;

		bool getValues( ZVectorArray& values, int sampleIndex=0 ) const;

		bool getValues( ZColorArray& values, int sampleIndex=0 ) const;
};

ostream&
operator<<( ostream& os, const ZAlembicProperty& object );

ZELOS_NAMESPACE_END

#endif

