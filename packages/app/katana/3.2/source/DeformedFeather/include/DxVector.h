#pragma once

// #include <cmath>
#include <math.h>
#include "DxMatrix.h"

namespace Dx
{
    class Vector
    {
        // variarble
    public:
        float x;
        float y;
        float z;

    public:
        Vector() // Default Construct
        {
            x = 0;
            y = 0;
            z = 0;
        }
        Vector(const Vector& v)
        {
            x = v.x;
            y = v.y;
            z = v.z;
        }
        Vector(const float& _x, const float& _y, const float& _z=0.0f)
        {
            x = _x;
            y = _y;
            z = _z;
        }

    public:
        Vector cross(const Vector& v) const
        {
            float _x = (y * v.z) - (z * v.y);
            float _y = (z * v.x) - (x * v.z);
            float _z = (x * v.y) - (y * v.x);
            return Vector(_x, _y, _z);
        }

        float dot(const Vector& v) const
        {
            return ( (x * v.x) + (y * v.y) + (z * v.z) );
        }

        float length()
        {
            float dotResult = (x * x) + (y * y) + (z * z);
            return sqrt(dotResult);
        }

        void normalize()
        {
            float len = length();
            x = x / len;
            y = y / len;
            z = z / len;
        }

        Vector operator*( const Matrix& m) const
        {
            float _x = x * m._00 + y * m._10 + z * m._20;
            float _y = x * m._01 + y * m._11 + z * m._21;
            float _z = x * m._02 + y * m._12 + z * m._22;
            return Vector(_x, _y, _z);
        }

        Vector operator+( const Vector& v) const
        {
            return Vector(x + v.x, y + v.y, z + v.z);
        }

        Vector operator-( const Vector& v) const
        {
            return Vector(x - v.x, y - v.y, z - v.z);
        }

        Vector operator/( const float& s ) const
        {
            return Vector(x / s, y / s, z / s);
        }

        Vector operator*( const float& s ) const
        {
            return Vector(x * s, y * s, z * s);
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
