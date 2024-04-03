//--------------//
// ZGlslOcean.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// 		   Jinhyuk Bae @ Dexter Studios					 //
// 		   Nayoung Kim @ Dexter Studios					 //
// last update: 2015.10.21                               //
//-------------------------------------------------------//

#ifndef _ZGlslOcean_h_
#define _ZGlslOcean_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlslOcean
{
	private:
		
		//=============
		// id
		//=============
		//
		GLuint		_programId;
		// attributes
		GLuint		_vertexId, _normalId, _textureId;
		// uniform
		GLuint		_uProjectionId, _uModelViewId, _uLightPosId;
		GLuint  	_uCameraPosId;
		// vertex buffer object
		GLuint		_vbo_verticesId, _vbo_indicesId, _vbo_normalId;
		// vertex array object
		GLuint 		_vaoId;

		bool _preprocessed;
		

	public:

		// shader program
		ZGlslProgram		_programOcean;

		// trimesh data
		ZTriMesh*			_triMeshPtr;
		ZVectorArray*		_normalPtr;
		ZFloatArray*		_foamPtr;
		ZArray<GLushort>	_vIndexArr;
		int				 	_numOfVtx;
		// camera
		ZPoint				_cameraPos;

		// matrix
 		ZMatrix 			_modelViewMat;
		ZMatrix 			_projectionMat;
		
		bool			 	_doGenGLBuffer;

	public:
	
		ZGlslOcean();
		~ZGlslOcean();

		ZGlslOcean* toGlslOcean() { return (ZGlslOcean*) this; }

		void setTriMeshPtr( ZTriMesh* triMesh );
		void setNormalPtr( ZVectorArray* normalPtr );
		void setFoamPtr( ZFloatArray* foamPtr );
		void setModelViewMatrix( const ZMatrix& modelViewMat );
		void setProjectionMatrix( const ZMatrix& projectionMat );

		void preprocess( const ZString& vsPath, const ZString& fsPath );
		void postprocess();
		void draw();

	private: 
		
		void reset();
		bool setShaders( const char* vertexShaderSrc, const char* fragmentShaderSrc );
		

//		GLint  _viewport[4];
//		ZGlTex3D _volumeData; // 3d volume data texture
//
//		ZPoint _center;
//		ZPoint _light;
//		ZPoint _eye;
//
//		ZScalarField3D* _density;

//	public:
//
//		ZGlslOcean();
//
//		//ZGlslOcean* toVolumeSmoke() { return (ZGlslOcean*) this; }
//
//		void draw();
//		void toggleDrawLightDir();
//
//		// control parameters
//		float _densityFactor;
//		float _brightness;
//		float _absorption;

//		int	  _numSamples;
//		int	  _numLightSamples;
//
//		bool _preprocessed;
//
//	public:
//
//		void preprocess( const ZScalarField3D& density );
//		void postprocess();
//
//	protected:
//
//		void setupOcean( const ZScalarField3D& density );
//		void setupFbo();
//		void setupProgram();
//
//		// Shader programs
//		ZGlslProgram    _programRaymarcher;
//
//		// frame buffer object
//		ZGlFbo			_fbo;	
//		ZGlRbo			_rbo;
//		ZGlTex2D		_tex;
//
//		// rayCasting 
//		void			drawBox(float w, float h, float d);
//		void			rayCasting();
//		void			renderTexturedQuad();
//
//		// draw light direction
//		bool			bDrawLightDir;
//		void			drawLightDir();



};

ZELOS_NAMESPACE_END

#endif

