//-------------//
// ZGlBitmap.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2016.07.04                               //
//-------------------------------------------------------//

#ifndef _ZGlBitmap_h_
#define _ZGlBitmap_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlBitmap
{
	public:

    	int            _width;				///< The image width.
		int            _height;				///< The image height.
		unsigned char* _pixel;				///< The pixel data.
		ZString        _windowTitle;		///< The title string to be displayed on the title bar.

	public:

		/// @brief The default constructor.
		/**
			Create an empty bitmap instance.
 		*/
		ZGlBitmap();

		/// @brief The class constructor.
		/**
			Create a new bitmap instance and initialize with the given size.
			@note The default value for the image depth is four.
			@param[in] width The image width.
			@param[in] height The image height.
 		*/
		ZGlBitmap( int width, int height );

		/// @brief The destroyer.
		/**
			Release all of the memory allocated in this instance.
 		*/
		virtual ~ZGlBitmap();

		/// @brief The re-initializer.
		/**
			Re-initialize the bitmap with the given parameters.
			@note The default value for the image depth is four.
			@param[in] width The image width.
			@param[in] height The image height.
			@return true if success and false otherwise.
		*/
		bool setSize( int width, int height );

		/// @brief The function for setting the window title.
		/**
			Set the title to be displayed on the title bar.
			@param[in] title The title string to be displayed on the title bar.
		*/
		void setWindowTitle( const char* title );

		/// @brief The number of pixels.
		/**
			Return the number of pixels for this instance.
			@return The number of pixels for this instance.
		*/
		int numPixels() const;

		/// @brief The total memory size.
		/**
			Return the total memory size for this instance.
			@return The total memory size for this instance.
		*/
		int size() const;

		/// @brief The width.
		/**
			Return the image width for this instance.
			@return The image width for this instance.
		*/
		int width() const;

		/// @brief The height.
		/**
			Return the image height for this instance.
			@return The image height for this instance.
		*/
		int height() const;

		/// @brief The pointer of the data.
		/**
			Return the pointer of the data.
			@return The pointer of the data.
		*/
		unsigned char* pointer() const;

		/// @brief The pointer of the data.
		/**
			Return the pointer of the data.
			@return The pointer of the data.
			@note This method would be used for static functions such as GLUT callbacks.
		*/
		static ZGlBitmap** bitmapPointer();

		/// @brief The index operator.
		/**
			Return the 1D array index from the given 3D indices.
			This is a utility routine for finding the index of an element in an 1D array.
			This method converts 3D cooridinates of an element into the index value that refers the element's value in the 1D array.
			@note No range checking is done for efficiency.
			@param[in] i The index of width.
			@param[in] j The index of height.
			@param[in] k The index of depth.
			@return The index in the single dimenstional array.
		*/
		int index( int i, int j, int k ) const;

		/// @brief The index operator.
		/**
			The index operator.
			Return the value of the element at the given index.
			@note No range checking is done for efficiency. Valid indices are 0 to size()-1.
			@param[in] i The index of width.
			@param[in] j The index of height.
			@param[in] k The index of depth.
			@return A const reference of the element.
		*/
		const unsigned char& operator()( const int& i, const int& j, const int& k ) const;

		/// @brief The index operator.
		/**
			The index operator.
			Return a reference to the element at the given index.
			@note No range checking is done for efficiency. Valid indices are 0 to size()-1.
			@param[in] i The index of width.
			@param[in] j The index of height.
			@param[in] k The index of depth.
			@return A reference to the indicated element.
		*/
		unsigned char& operator()( const int& i, const int& j, const int& k );

		// OpenGL drawing function
		void display();

		// static method used for glut callbacks
		static void keyboard( unsigned char key, int x, int y );

	    // static method used for glut callbacks
    	static void draw();
};

inline int
ZGlBitmap::index( int i, int j, int k ) const
{
	return ( 4*(i+j*_width)+k );
}

inline const unsigned char&
ZGlBitmap::operator()( const int& i, const int& j, const int& k ) const
{
	return _pixel[ 4*(i+j*_width)+k ];
}

inline unsigned char&
ZGlBitmap::operator()( const int& i, const int& j, const int& k )
{
	return _pixel[ 4*(i+j*_width)+k ];
}

ostream&
operator<<( ostream& os, const ZGlBitmap& object );

ZELOS_NAMESPACE_END

#endif

