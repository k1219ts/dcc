//-----------------------//
// ZLinearSystemSolver.h //
//-------------------------------------------------------//
// author: Jaegwang Lim @ Dexter Studios                 //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2016.06.20                               //
//-------------------------------------------------------//

#ifndef _ZLinearSystemSolver_h_
#define _ZLinearSystemSolver_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZLinearSystemSolver
{
    private:  

        ZFloatArray  t, r, z, r0, z0, p, Ap;        
        
    public:
    
        // for providing informations.
        int maxIterations;
        int curIterations;

    public:
        void CG       ( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter=300 );
        void PCG      ( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter=300 );
        void Jacobian ( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter=300 ){};
        void Gaussian ( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter=300 ){};
};

inline void 
ZLinearSystemSolver::CG( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter )
{    
    Multiply(A, x, t);
    Subtract(b, t, r);
    Equal(r, p);
    
    float denom(0.0);
    float rsold(0.0);
    Multiply(r, r, rsold);
    
    if(sqrt(rsold) < 1e-6) return;
    
    const int numIter = std::min(maxIter, (int)b.size()*2);
    
    int it = 0;
    for(; it<numIter; it++)
    {
        Multiply(A, p, Ap);
        
        float alpha(0.0);
        Multiply(p, Ap, denom);
        alpha = rsold / denom;
        
        Multiply(p, alpha, t);
        Add(x, t, x);
        
        Multiply(Ap, alpha, t);
        Subtract(r, t, r);
        
        float rsnew(0.0);
        Multiply(r, r, rsnew);
        
        if(sqrt(rsnew) < 1e-6) break;
        
        Multiply(p, rsnew/rsold, t);
        Add(r, t, p);
        
        rsold = rsnew;
    }   
    
    maxIterations = numIter;
    curIterations = it;    
}

inline void 
ZLinearSystemSolver::PCG( const ZSparseMatrix<float>& A, ZFloatArray& x, const ZFloatArray& b, const int maxIter )
{    
    Multiply(A, x, t);
    Subtract(b, t, r);
    
    float denom(0.0);
    
    InverseDiagonalMultiply(A, r, z);    
    Equal(z, p);       
    
    float rsold(0.0);
    Multiply(z, z, rsold);
    
    if(sqrt(rsold) < 1e-6) return;    
    
    const int numIter = std::min(maxIter, (int)b.size()*2);
    
    int it =0;
    for(;it<numIter; it++)
    {
        Equal(z, z0);
        Equal(r, r0);
        
        float alpha(0.0);
        Multiply(r, z, alpha );        
        Multiply(A, p, Ap    );
        Multiply(p, Ap, denom);        
        alpha = alpha / denom;
        
        Multiply(p, alpha, t);
        Add(x, t, x);
        
        Multiply(Ap, alpha, t);
        Subtract(r, t, r);
        
        float rs(0.0);
        Multiply(r, r, rs);
        if(sqrt(rs) < 1e-06) break;
        
        InverseDiagonalMultiply(A, r, z);
        
        float beta(0.0);
        Multiply(z , r , beta );
        Multiply(z0, r0, denom);
        beta = beta / denom;
        
        Multiply(p, beta, t);
        Add(z, t, p);
    }
    
    maxIterations = numIter;
    curIterations = it;
}

ZELOS_NAMESPACE_END

#endif

