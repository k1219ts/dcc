//--------------------//
// ZEquationZSolvers.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2017.11.29                               //
//-------------------------------------------------------//

#ifndef _ZEquationZSolvers_h_
#define _ZEquationZSolvers_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

// Linear equation
// ax + b = 0
inline void ZSolveLinearEqn( float a, float b, float& x )
{
    if( a == 0 )
    {
        x = NAN;
    }

    x = (float)( -(double)b / (double)a );
}

// Quadratic equation for real roots
// ax^2 + bx + c = 0
inline int ZSolveQuadraticEqn( float a, float b, float c, float x[2] )
{
    if( ZAlmostZero(a) ) // linear case
    {
        ZSolveLinearEqn( b, c, x[0] );
        x[1] = x[0];

        return 1;
    }

    const double D = b*b - 4*a*c;
    const double _2a = 1 / (2*a);

    if( D==0 ) // one real root
    {
        x[0] = x[1] = (float)( -b * _2a );

        return 1;
    }
    else // two roots
    {
        if( D>0 ) // two real roots
        {
            const double s = sqrt(D);

            x[0] = (float)( ( -b - s ) * _2a );
            x[1] = (float)( ( -b + s ) * _2a );

            return 2;
        }
        else // D<0: no roots
        {
            x[0] = x[1] = NAN;

            return 0;
        }
    }
}

// Quadratic equation for complex roots
// ax^2 + bx + c = 0
inline int ZSolveQuadraticEqn( float a, float b, float c, ZComplex x[2] )
{
    if( ZAlmostZero(a) ) // linear case
    {
        ZSolveLinearEqn( b, c, x[0].r );
        x[1] = x[0];

        return 1;
    }

    const double D = b*b - 4*a*c;
    const double _2a = 1 / (2*a);

    if( D==0 ) // one real root
    {
        x[0] = x[1] = (float)( -b * _2a );

        return 1;
    }
    else // two roots
    {
        if( D>0 ) // two real roots
        {
            const double s = sqrt(D);

            x[0] = (float)( ( -b - s ) * _2a );
            x[1] = (float)( ( -b + s ) * _2a );
        }
        else // D<0: two complex roots
        {
            x[0].r = (float)( -b * _2a );
            x[0].i = (float)( sqrt( ZAbs(D) ) * _2a );

            x[1] = x[0].conjugated();
        }

        return 2;
    }
}

// Cubic equation for real roots
// ax^3 + bx^2 + cx + d = 0
inline int ZSolveCubicEqn( float a, float b, float c, float d, float x[3] )
{
    if( a==0 && b==0 ) // linear case
    {
        ZSolveLinearEqn( c,d, x[0] );
        x[1] = x[2] = x[0];
        return 1;
    }

    if( a==0 ) // quadratic case
    {
        ZSolveQuadraticEqn( b,c,d, &x[0] );
        x[2] = x[0];
        return 2;
    }

    const double f = ((3.0*c/a)-((b*b)/(a*a)))/3.0;
    const double g = (((2.0*b*b*b)/(a*a*a))-((9.0*b*c)/(a*a))+(27.0*d/a))/27.0;
    const double h = ((g*g)/4.0+(f*f*f)/27.0);

    if( f==0 && g==0 && h==0 ) // all three roots are real and equal
    {
        if( d/a >= 0 )
        {
            x[0] = x[1] = x[2] = (float)( -pow(d/a,1/3.0) );
            return 1;
        }
        else
        {
            x[0] = x[1] = x[2] = (float)( pow(-d/a,1/3.0) );
            return 1;
        }
    }
    else if( h<=0 ) // all three roots are real
    {
        const double i = sqrt(((g*g)/4.0)-h);
        const double j = pow(i,1/3.0);
        const double k = acos(-g/(2.0*i));
        const double L = -j;
        const double M = cos(k/3.0);
        const double N = sqrt(3)*sin(k/3.0);
        const double P = -b/(3.0*a);

        x[0] = (float)( 2.0*j*cos(k/3.0)-(b/(3.0*a)) );
        x[1] = (float)( L*(M+N)+P );
        x[2] = (float)( L*(M-N)+P );

        if( ZAlmostSame( x[0],x[1] ) ) { return 2; }
        if( ZAlmostSame( x[1],x[2] ) ) { return 2; }
        if( ZAlmostSame( x[2],x[0] ) ) { return 2; }

        return 3;
    }
    else if( h>0 ) // one real root and two complex roots
    {
        const double u = -(g/2.0)+sqrt(h);
        const double U = ZSign(u)*pow(ZAbs(u),1/3.0);
        const double v = -(g/2.0)-sqrt(h);
        const double V = ZSign(v)*pow(ZAbs(v),1/3.0);

        x[0] = (float)( (U+V)-(b/(3.0*a)) );
        x[1] = NAN;
        x[2] = NAN;

        return 1;
    }

    return 0;
}

// Cubic equation for complex roots
// ax^3 + bx^2 + cx + d = 0
inline int ZSolveCubicEqn( float a, float b, float c, float d, ZComplex x[3] )
{
    if( a==0 && b==0 ) // linear case
    {
        ZSolveLinearEqn( c,d, x[0].r );
        x[1] = x[2] = x[0];
        return 1;
    }

    if( a==0 ) // quadratic case
    {
        ZSolveQuadraticEqn( b,c,d, &x[0] );
        x[2] = x[0];
        return 2;
    }

    const double f = ((3.0*c/a)-((b*b)/(a*a)))/3.0;
    const double g = (((2.0*b*b*b)/(a*a*a))-((9.0*b*c)/(a*a))+(27.0*d/a))/27.0;
    const double h = ((g*g)/4.0+(f*f*f)/27.0);

    if( f==0 && g==0 && h==0 ) // all three roots are real and equal
    {
        if( d/a >= 0 )
        {
            x[0] = x[1] = x[2] = (float)( -pow(d/a,1/3.0) );
            return 1;
        }
        else
        {
            x[0] = x[1] = x[2] = (float)( pow(-d/a,1/3.0) );
            return 1;
        }
    }
    else if( h <= 0 ) // all three roots are real
    {
        const double i = sqrt(((g*g)/4.0)-h);
        const double j = pow(i,1/3.0);
        const double k = acos(-g/(2.0*i));
        const double L = -j;
        const double M = cos(k/3.0);
        const double N = sqrt(3)*sin(k/3.0);
        const double P = -b/(3.0*a);

        x[0] = (float)( 2.0*j*cos(k/3.0)-(b/(3.0*a)) );
        x[1] = (float)( L*(M+N)+P );
        x[2] = (float)( L*(M-N)+P );

        if( ZAlmostSame( x[0].r,x[1].r ) ) { return 2; }
        if( ZAlmostSame( x[1].r,x[2].r ) ) { return 2; }
        if( ZAlmostSame( x[2].r,x[0].r ) ) { return 2; }

        return 3;
    }
    else if( h > 0 ) // one real root and two complex roots
    {
        const double u = -(g/2.0)+sqrt(h);
        const double U = ZSign(u)*pow(ZAbs(u),1/3.0);
        const double v = -(g/2.0)-sqrt(h);
        const double V = ZSign(v)*pow(ZAbs(v),1/3.0);

        x[0] = (float)( (U+V)-(b/(3.0*a)) );
        x[1] = ZComplex( (float)( -(U+V)/2.0-(b/(3.0*a)) ), (float)( (U-V)*sqrt(3)*0.5 ) );
        x[2] = ZComplex( (float)( -(U+V)/2.0-(b/(3.0*a)) ), (float)(-(U-V)*sqrt(3)*0.5 ) );

        return 3;
    }

    return 0;
}

// Simulataneous equation
// ax + by + c = 0
// px + qy + r = 0
inline bool ZSolveSimultaneousEqn( float a, float b, float c, float p, float q, float r, float& x, float& y )
{
    const float d = a*q - b*p;

    if( ZAlmostZero(d) )
    {
        x = y = Z_LARGE;
        return false;
    }

    x = ( b*r - c*q ) / d;

    if( ZAlmostZero(q) )
    {
        y = Z_LARGE;
        return false;
    }

    y = -1/q * ( p*x + r );

    return true;
}

ZELOS_NAMESPACE_END

#endif

