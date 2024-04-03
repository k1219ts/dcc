#pragma once

#include <cmath>
#include "DxMatrix.h"
#include "DxVector.h"

namespace Dx
{
    class Point
    {
        // variarble
    public:
        float x;
        float y;
        float z;

    public:
        Point() // Default Construct
        {
            x = 0;
            y = 0;
            z = 0;
        }
        Point(const Point& v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }
        Point(const float& _x, const float& _y, const float& _z=0.0f)
        {
            x = _x;
            y = _y;
            z = _z;
        }

    public:
        Point cross(const Point& v) const
        {
            float _x = (y * v.z) - (z * v.y);
            float _y = (z * v.x) - (x * v.z);
            float _z = (x * v.y) - (y * v.x);
            return Point(_x, _y, _z);
        }

        float dot(const Point& v) const
        {
            return ( (x * v.x) + (y * v.y) + (z * v.z) );
        }

        Point cross(const Vector& v) const
        {
            float _x = (y * v.z) - (z * v.y);
            float _y = (z * v.x) - (x * v.z);
            float _z = (x * v.y) - (y * v.x);
            return Point(_x, _y, _z);
        }

        float dot(const Vector& v) const
        {
            return ( (x * v.x) + (y * v.y) + (z * v.z) );
        }

        float length()
        {
            float dotResult = (x * x) + (y * y) + (z * z);
            return sqrtf(dotResult);
        }

        void normalize()
        {
            float len = length();
            x = x / len;
            y = y / len;
            z = z / len;
            // if(isnan(x) || isnan(y) || isnan(z))
            // {
            //     std::cout << "length" << len << std::endl;
            // }
        }

        Point operator*( const Matrix& m) const
        {
            float _x = x * m._00 + y * m._10 + z * m._20 + m._30;
            float _y = x * m._01 + y * m._11 + z * m._21 + m._31;
            float _z = x * m._02 + y * m._12 + z * m._22 + m._32;
            return Point(_x, _y, _z);
        }

        Point operator+( const Point& v) const
        {
            return Point(x + v.x, y + v.y, z + v.z);
        }

        Point operator-( const Point& v) const
        {
            return Point(x - v.x, y - v.y, z - v.z);
        }

        Point operator/( const float& s ) const
        {
            return Point(x / s, y / s, z / s);
        }

        Point operator*( const float& s ) const
        {
            return Point(x * s, y * s, z * s);
        }

        float& operator[](const int& i)
        {
            switch(i)
            {
                default:
                case 0: { return x; }
                case 1: { return y; }
                case 2: { return z; }
            }
        }

        bool isNan(std::string debugString)
        {
            if(isnan(x) || isnan(y) || isnan(z))
            {
                std::cout << debugString << std::endl;
                Print();
                return true;
            }

            return false;
        }

        void Print()
        {
            std::cout << "[" << x << " " << y << " " << z << "]" << std::endl;
        }
    };
}
