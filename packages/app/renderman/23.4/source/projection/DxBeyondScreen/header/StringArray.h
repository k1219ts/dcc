#ifndef _BS_StringArray_h_
#define _BS_StringArray_h_

#include <BeyondScreen.h>

BS_NAMESPACE_BEGIN

class StringArray : public vector<String>
{
    private:

        typedef vector<String> parent;

    public:

        StringArray()
        : vector<String>()
        {
            // nothing to do
        }

        void clear()
        {
            parent::clear();
        }

        String& first()
        {
            return parent::front();
        }

        const String& first() const
        {
            return parent::front();
        }

        String& last()
        {
            return parent::back();
        }

        const String& last() const
        {
            return parent::back();
        }

        void append( const String& str )
        {
            parent::push_back( str );
        }

        int setByTokenizing( const String& str, const String& delimiter )
        {
            parent::clear();

            string::size_type lastPos = str.find_first_not_of( delimiter, 0 );
            string::size_type pos = str.find_first_of( delimiter, lastPos );

            while( str.npos != pos || str.npos != lastPos )
            {
                parent::push_back( str.substr( lastPos, pos - lastPos ) );
                lastPos = str.find_first_not_of( delimiter, pos );
                pos = str.find_first_of( delimiter, lastPos );
            }

            return (int)parent::size();
        }

        int length() const
        {
            return (int)parent::size();
        }
};

BS_NAMESPACE_END

#endif

