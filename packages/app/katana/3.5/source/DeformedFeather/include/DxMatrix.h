#pragma once

namespace Dx
{
    class Matrix
    {
    public:
        union
        {
            struct
            {
                float _00, _01, _02, _03;
                float _10, _11, _12, _13;
                float _20, _21, _22, _23;
                float _30, _31, _32, _33;
            };
            float data[4][4];
        };

    public:
        Matrix()
        {
            _00=1.f; _01=0.f; _02=0.f; _03=0.f;
            _10=0.f; _11=1.f; _12=0.f; _13=0.f;
            _20=0.f; _21=0.f; _22=1.f; _23=0.f;
            _30=0.f; _31=0.f; _32=0.f; _33=1.f;
        }

        Matrix(const float& m00, const float& m01, const float& m02, const float& m03,
               const float& m10, const float& m11, const float& m12, const float& m13,
               const float& m20, const float& m21, const float& m22, const float& m23,
               const float& m30, const float& m31, const float& m32, const float& m33)
       {
            _00=m00; _01=m01; _02=m02; _03=m03;
            _10=m10; _11=m11; _12=m12; _13=m13;
            _20=m20; _21=m21; _22=m22; _23=m23;
            _30=m30; _31=m31; _32=m32; _33=m33;
       }

       double det()
       {
           return double(
               _03 * _12 * _21 * _30 - _02 * _13 * _21 * _30 -
                _03 * _11 * _22 * _30 + _01 * _13 * _22 * _30 +
                _02 * _11 * _23 * _30 - _01 * _12 * _23 * _30 -
                _03 * _12 * _20 * _31 + _02 * _13 * _20 * _31 +
                _03 * _10 * _22 * _31 - _00 * _13 * _22 * _31 -
                _02 * _10 * _23 * _31 + _00 * _12 * _23 * _31 +
                _03 * _11 * _20 * _32 - _01 * _13 * _20 * _32 -
                _03 * _10 * _21 * _32 + _00 * _13 * _21 * _32 +
                _01 * _10 * _23 * _32 - _00 * _11 * _23 * _32 -
                _02 * _11 * _20 * _33 + _01 * _12 * _20 * _33 +
                _02 * _10 * _21 * _33 - _00 * _12 * _21 * _33 -
                _01 * _10 * _22 * _33 + _00 * _11 * _22 * _33
           );
       }

       Matrix inverse()
       {
            const double _det = 1.0 / ( det() + 1e-30 );

            const double m00 = _00, m01 = _01, m02 = _02, m03 = _03;
            const double m10 = _10, m11 = _11, m12 = _12, m13 = _13;
            const double m20 = _20, m21 = _21, m22 = _22, m23 = _23;
            const double m30 = _30, m31 = _31, m32 = _32, m33 = _33;

            float __00 = (float)(  ( m11*(m22*m33-m23*m32) - m12*(m21*m33-m23*m31) + m13*(m21*m32-m22*m31) ) * _det );
            float __01 = (float)( -( m01*(m22*m33-m23*m32) - m02*(m21*m33-m23*m31) + m03*(m21*m32-m22*m31) ) * _det );
            float __02 = (float)(  ( m01*(m12*m33-m13*m32) - m02*(m11*m33-m13*m31) + m03*(m11*m32-m12*m31) ) * _det );
            float __03 = (float)( -( m01*(m12*m23-m13*m22) - m02*(m11*m23-m13*m21) + m03*(m11*m22-m12*m21) ) * _det );

            float __10 = (float)( -( m10*(m22*m33-m23*m32) - m12*(m20*m33-m23*m30) + m13*(m20*m32-m22*m30) ) * _det );
            float __11 = (float)(  ( m00*(m22*m33-m23*m32) - m02*(m20*m33-m23*m30) + m03*(m20*m32-m22*m30) ) * _det );
            float __12 = (float)( -( m00*(m12*m33-m13*m32) - m02*(m10*m33-m13*m30) + m03*(m10*m32-m12*m30) ) * _det );
            float __13 = (float)(  ( m00*(m12*m23-m13*m22) - m02*(m10*m23-m13*m20) + m03*(m10*m22-m12*m20) ) * _det );

            float __20 = (float)(  ( m10*(m21*m33-m23*m31) - m11*(m20*m33-m23*m30) + m13*(m20*m31-m21*m30) ) * _det );
            float __21 = (float)( -( m00*(m21*m33-m23*m31) - m01*(m20*m33-m23*m30) + m03*(m20*m31-m21*m30) ) * _det );
            float __22 = (float)(  ( m00*(m11*m33-m13*m31) - m01*(m10*m33-m13*m30) + m03*(m10*m31-m11*m30) ) * _det );
            float __23 = (float)( -( m00*(m11*m23-m13*m21) - m01*(m10*m23-m13*m20) + m03*(m10*m21-m11*m20) ) * _det );

            float __30 = (float)( -( m10*(m21*m32-m22*m31) - m11*(m20*m32-m22*m30) + m12*(m20*m31-m21*m30) ) * _det );
            float __31 = (float)(  ( m00*(m21*m32-m22*m31) - m01*(m20*m32-m22*m30) + m02*(m20*m31-m21*m30) ) * _det );
            float __32 = (float)( -( m00*(m11*m32-m12*m31) - m01*(m10*m32-m12*m30) + m02*(m10*m31-m11*m30) ) * _det );
            float __33 = (float)(  ( m00*(m11*m22-m12*m21) - m01*(m10*m22-m12*m20) + m02*(m10*m21-m11*m20) ) * _det );

            Matrix m = Matrix(__00, __01, __02, __03,
                                __10, __11, __12, __13,
                                __20, __21, __22, __23,
                                __30, __31, __32, __33);

            return m;
       }

       void Print()
       {
           std::cout << "===== Matrix =====" << std::endl;
           std::cout << _00 << ", " << _01 << ", " << _02 << ", " << _03 << std::endl;
           std::cout << _10 << ", " << _11 << ", " << _12 << ", " << _13 << std::endl;
           std::cout << _20 << ", " << _21 << ", " << _22 << ", " << _23 << std::endl;
           std::cout << _30 << ", " << _31 << ", " << _32 << ", " << _33 << std::endl;
           std::cout << "===== Matrix =====" << std::endl;
       }
    };
}
