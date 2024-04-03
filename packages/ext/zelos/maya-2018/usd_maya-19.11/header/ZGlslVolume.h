//---------------//
// ZGlslVolume.h //
//-------------------------------------------------------//
// author: Wanho Choi @ Dexter Studios                   //
// last update: 2015.10.07                               //
//-------------------------------------------------------//

#ifndef _ZGlslVolume_h_
#define _ZGlslVolume_h_

#include <ZelosBase.h>

ZELOS_NAMESPACE_BEGIN

class ZGlslVolume
{
	private:

		GLint  _viewport[4];
		ZGlTex3D _volumeData; // 3d volume data texture

		ZPoint _center;
		ZPoint _light;
		ZPoint _eye;

ZScalarField3D* _density;

	public:

		ZGlslVolume();

		ZGlslVolume* toVolumeSmoke() { return (ZGlslVolume*) this; }

		void draw();
		void toggleDrawLightDir();

		// control parameters
		float _densityFactor;
		float _brightness;
		float _absorption;
		int	  _numSamples;
		int	  _numLightSamples;

bool _preprocessed;

	public:

		void preprocess( const ZScalarField3D& density );
		void postprocess();

	protected:

		void setupVolume3D( const ZScalarField3D& density );
		void setupFbo();
		void setupProgram();

		// Shader programs
		ZGlslProgram    _programRaymarcher;

		// frame buffer object
		ZGlFbo			_fbo;	
		ZGlRbo			_rbo;
		ZGlTex2D		_tex;

		// rayCasting 
		void			drawBox(float w, float h, float d);
		void			rayCasting();
		void			renderTexturedQuad();

		// draw light direction
		bool			bDrawLightDir;
		void			drawLightDir();
};

ZELOS_NAMESPACE_END

#endif

