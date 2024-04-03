//--------------//
// ZParticles.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// 		   Jinhyuk Bae @ Dexter Studios					 //
// last update: 2017.06.09                               //
//-------------------------------------------------------//

#ifndef _ZParticles_h_
#define _ZParticles_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

/// @brief A class for a 3D particles which can have arbitrary attributes.
/**
	This class provides an implementation of a 3D particles.
	Numerous convenient operators are provided to help with the manipulation of particles.
*/
class ZParticles
{
	private:

		int _numAttributes;						///< The number of attributes.
		int _numParticles;						///< The number of particles.
		int _numAllocated;						///< The real number of particles allocated on memory.

		int    _groupId;						///< The group integer ID of the particles.
		ZColor _groupColor;						///< The group color ID of the particles.

		ZBoundingBox _aabb;                     ///< The axis-aligned bounding box

		ZIntArray             _dataType;		///< The data typa of each attribute.
		ZIntArray             _dataSize;		///< The data size of each attribute.
		ZStringArray          _attrName;		///< The name of each attribute.
		std::map<ZString,int> _nameToIndex;		///< The mapping from the name to the attribute index.
		std::vector<char*>    _data;			///< The pointer for the data of each attribute.

	public:

		/**
			Default constructor.
			Create a new empty particle set.
		*/
		ZParticles();

		/**
			Copy constructor.
			Create a new particles and initialize it ot the same content as the given particles.
		*/
		ZParticles( const ZParticles& source );

		/**
			Class constructor.
			Create a new particles and initialize it from a file.
			@param[in] filePathName The file path and name to load from it.
		*/
		ZParticles( const char* filePathName );

		/**
			Destructor.
			Release all of the memory allocated in this instance.
		*/
		virtual ~ZParticles();

		/**
			Reset the particles.
		*/
		void reset();

		/**
			Check if two particles are compatible in attributes.
		*/
		bool checkCompatible( const ZParticles& source ) const;

		/**
			The assignement operator.
			Copy all of the contents from the given particles into this one.
			@param[in] other The particles to copy from.
			@return A reference of the particles after being copied.
		*/
		ZParticles& operator=( const ZParticles& other );

		/**
			Return the number of particles.
			@return The number of particles.
		*/
		int numParticles() const;

		/**
			Return the number of attributes.
			@return The number of attributes.
		*/
		int numAttributes() const;

		/**
			Return the data type ID of the attribute at the given index.
			@param[in] i The index of the attribute being queried.
			@return The integer type ID of the data type.
		*/
		int dataType( const int& i ) const;

		/**
			Return the data size (bytes) of the attribute at the given index.
			@param[in] i The index of the attribute being queried.
			@return The size of the data type.
		*/
		int dataSize( const int& i ) const;

		/**
			Return the name of the attribute at the given index.
			@param[in] i The index of the attribute being queried.
			@return The name of the attribute being queried.
		*/
		ZString attributeName( const int& i ) const;

		/**
			Return the name of the attribute by the given name.
			@param[in] name The name of the attribute being queried.
			@return The index of the attribute being queried. \n
			It will return -1 if the given name of attribute does not exist.
		*/
		int attributeIndex( const char* name ) const;

		/**
			Add a new attribute to this particles.
			@param[in] name The name of the attribute being added.
			@param[in] dataType The data type of the attribute.
			@return true if success and false otherwise.
		*/
		bool addAttribute( const char* name, ZDataType::DataType zDataType );

		/**
			Delete the attribute by the given name.
			@param[in] name The name of the attribute being deleted.
			@return true if success and false otherwise.
		*/
		bool deleteAttribute( const char* name );

		/**
			Add a number of particles to this particles instance.
			@param[in] numToAdd The number of particles being added.
		*/
		bool addParticles( const int& numToAdd );

		/**
			Add given particles to the end of this one.
			@param[in] particles The given particles being added at the end.
			@return true if two particles have same attributes and false otherwise.
		*/
		bool append( const ZParticles& particles );

		/**
			Remove the particles listed in the given array.
			All particles following the removed element are shifted toward the first element.
			@param[in] indicesToBeDeleted The array of indices to be deleted.
			@return The final number of the particles.
		*/
		int remove( const ZIntArray& indicesToBeDeleted );

		/**
			Return the pointer to the i-th particle of the given name of the attribute.
			@param[in] name The name of attribute being queried.
			@param[in] i The particle index being queried.
			@return The pointer to the data.
		*/
		void* data( const char* attrName, const int& i=0 ) const;

		/**
			Return the pointer to the i-th particle of the given name of the attribute.
			@param[in] attrIndex attribute index 
			@param[in] i The particle index being queried.
			@return The pointer to the data.
		*/
		void* data( const int& attrIndex, const int& i=0 ) const;

		/**
			Compute the axis-aligned bounding box of the particles.
			@param[in] attrName The attribute name 
			@param[in] useOpenMP A flag for using OpenMP or not while performaxg this function.
			@return True if success, False if failed.
		*/
		bool computeBoundingBox( const char* attrName="position", bool useOpenMP=true ); 

		/**
			Return the group ID of the particles.
			@return The group ID of the particles.
		*/
		int groupId() const;

		/**
			Return the group color of the particles.
			@return The group color of the particles.
		*/
		ZColor groupColor() const;

		/**
			Return the axis-aligned bounding box of the particles.
			@return The axis-aligned bounding box of the particles.
		*/
		ZBoundingBox boundingBox() const;

		/**
 			Return the minimum magnitude of the given attribute name.
			@param[in] name The name of attribute being queried.
			@param[in] useOpenMP A flag for using OpenMP or not while performing this function.
 			@return the minimum magnitude of the given attribute name.
 		*/
		float minMagnitude( const char* attrName, bool useOpenMP=true ) const;

		/**
 			Return the maximum magnitude of the given attribute name.
			@param[in] name The name of attribute being queried.
			@param[in] useOpenMP A flag for using OpenMP or not while performaxg this function.
 			@return the maximum magnitude of the given attribute name.
 		*/
		float maxMagnitude( const char* attrName, bool useOpenMP=true ) const;

		/**
			Save the data of the particles into a file.
			@param[in] filePathName The file path and name to save into it.
			@return True if success, False if failed.
		*/
		bool save( const char* filePathName ) const;

		/**
			Load the data of the particles from a file.
			@param[in] filePathName The file path and name to load from it.
			@return True if success, False if failed.
		*/
		bool load( const char* filePathName );

	private:

		bool _save_zelos( const char* filePathName ) const;
		bool _save_classicBGEO( const char* filePathName ) const;
		bool _save_alembic( const char* filePathName ) const;
		bool _save_bifrost( const char* filePathName ) const;

		bool _load_zelos( const char* filePathName );
		bool _load_classicBGEO( const char* filePathName );
		bool _load_alembic( const char* filePathName );
		bool _load_bifrost( const char* filePathName );
};

/**
	Print the information of the particles.
	@param[in] os The output stream object.
	@param[in] particles The particles to print.
	@return The output stream object.
*/
ostream&
operator<<( ostream& os, const ZParticles& particles );

ZELOS_NAMESPACE_END

#endif

