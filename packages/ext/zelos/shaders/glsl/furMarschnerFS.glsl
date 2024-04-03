//------------------//
// furMarschnerFS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

// from vertex shader
varying vec4      shadowCoordReadOnly;
varying float     sin_theta_r;
varying float     sin_theta_i;
varying float     cos_phi;
varying float     cos_theta_i;
varying float     cos_phi_H;

// from application
uniform int       useSelfshadow;
uniform vec2      viewport;
uniform float     diffuseLongitudeFalloff;
uniform float     diffuseAzimuthalFalloff;
uniform float     scaleDiffuse;
uniform float     scaleR;
uniform float     scaleTT;
uniform float     scaleTRT;
uniform float     transmittance;

// from application (textures)
uniform sampler2D depthMap;
uniform sampler2D opacityMap1;
uniform sampler2D opacityMap2;
uniform sampler2D lookupM;      // Mr, Mtt, Mtrt[RGB], normalized cos_theta_D
uniform sampler2D lookupN;      // Ntt[RGB], Nr[A]
uniform sampler2D lookupNTRT;   // Ntrt[RGBA], A: no use

float getOp( vec2 uv, float z, float z0 )
{
	float opacity = 0.0;

	// Get average of 5 by 5 adjacent pixels.
	vec4 op1 = vec4( 0.0 );
	for( int i=-2; i <=2 ; i++ )
	for( int j=-2; j <=2 ; j++ )
	{{
		vec2 sample;
		sample.x = uv.x - i * ( 1.0 / viewport.x );
		sample.y = uv.y - j * ( 1.0 / viewport.y );
		op1 += texture2D( opacityMap1, sample );
	}}
	op1 /= vec4( 25.0 );

	float d0 = ( 1.0- z0 ) / 15.0;   // 4+1 layers
	vec4  di = vec4( d0, 2.0*d0, 3.0*d0, 4.0*d0 );
	if     ( z >= z0+di.w ) {                               opacity = op1.w; }
	else if( z >= z0+di.z ) { float a = (z-(z0+di.z))/di.w; opacity = mix(op1.z, op1.w, a); }
	else if( z >= z0+di.y ) { float a = (z-(z0+di.y))/di.z; opacity = mix(op1.y, op1.z, a); }
	else if( z >= z0+di.x ) { float a = (z-(z0+di.x))/di.y; opacity = mix(op1.x, op1.y, a); }
	else if( z >= z0      ) { float a = (z- z0      )/di.x; opacity = mix(0.0,   op1.x, a); }

	return opacity;
}

void main() 
{
	//////////////////////
	// deep opacity map //

	vec4 shadowCoord = shadowCoordReadOnly;
	shadowCoord     /= shadowCoord.w;				// perspective divide
	shadowCoord      = 0.5*(shadowCoord+1.0);		// bias matrix: [-1,1]=>[0,1]
	vec2 uv          = shadowCoord.xy;
	float z          = shadowCoord.z;
	float z0         = texture2D( depthMap, uv ).x;
	float opacity    = getOp( uv, z, z0 );

	/////////////////////
	// Marschner model //

	// M: longitudinal scattering functions
	// sin_theta_i: [-pi,+pi):[-1,+1) => [-pi/2,+pi/2):[-0.5,+0.5) => s:[0,1)
	// sin_theta_r: [-pi,+pi):[-1,+1) => [-pi/2,+pi/2):[-0.5,+0.5) => t:[0,1) and flip
	vec2 lookupCoord1 = vec2( 0.5*sin_theta_i+0.5, 1.0-(0.5*sin_theta_r+0.5) );
	vec4 M            = texture2D( lookupM, lookupCoord1 );

	// N: azimuthal scattering functions
	// cos_phi    : [-pi,+pi):[-1,+1) => [-pi/2,+pi/2):[-0.5,+0.5) => s:[0,1)
	// cos_theta_d: already normalized to [0,1), and just flip
	vec2 lookupCoord2 = vec2( 0.5*cos_phi+0.5, 1.0-M.a );
	vec4 N     = texture2D( lookupN,    lookupCoord2 );
	vec4 N_TRT = texture2D( lookupNTRT, lookupCoord2 );

	/////////////
	// ambient //

	vec4 ambient = gl_LightSource[0].ambient * gl_Color + gl_LightModel.ambient * gl_Color;
	vec3 H = ambient.xyz;

	/////////////
	// diffuse //

	vec3 diffuse = gl_Color.xyz * mix( 1.0, cos_theta_i, diffuseLongitudeFalloff ) * mix( 1.0, cos_phi_H, diffuseAzimuthalFalloff );
	H += diffuse * scaleDiffuse * gl_LightSource[0].diffuse.rgb;

	//////////////
	// specular //

	vec3 spec;
	spec  = vec3( M.x * N.a       * scaleR   );			// Mr   * Nr
	spec += vec3( M.y * N.xyz     * scaleTT  );			// Mtt  * Ntt
	spec += vec3( M.z * N_TRT.xyz * scaleTRT );			// Mtrt * Ntrt
	H += spec * gl_LightSource[0].specular.rgb;

	/////////////////
	// self-shadow //

	if( useSelfshadow == 1 )
	{
		H *= exp( -transmittance * opacity ); 
	}

	//////////////////
	// final result //

	gl_FragColor = vec4( H, gl_Color.a );
}

