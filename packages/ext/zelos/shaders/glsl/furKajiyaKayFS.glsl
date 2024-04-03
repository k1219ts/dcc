//------------------//
// furKajiyaKayFS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

// from vertex shader
varying vec3      lightDir;
varying vec3      eyeDir;
varying vec3      hairDir;
varying vec4      shadowCoordReadOnly;

// from application
uniform int       useSelfshadow;
uniform int       useMRT;
uniform vec2      viewport;
uniform float     transmittance;

// from application (textures)
uniform sampler2D depthMap;
uniform sampler2D opacityMap1;
uniform sampler2D opacityMap2;

float getOp( vec2 uv, float z, float z0 )
{
	float opacity = 0.0;

	vec4 op1 = vec4( 0.0 );

	// Get average of 5 by 5 adjacent pixels.
	for( int i=-2; i<=2 ; i++ )
	for( int j=-2; j<=2 ; j++ )
	{{
		vec2 sample;
		sample.x = uv.x - i * ( 1.0 / viewport.x );
		sample.y = uv.y - j * ( 1.0 / viewport.y );
		op1 += texture2D( opacityMap1, sample );
	}}
	op1 /= vec4( 25.0 );

	float d0 = (1.0-z0)/15.0;	// 4+1 layers
	vec4  di = vec4( d0, 2.0*d0, 3.0*d0, 4.0*d0 );

	if     ( z >= z0+di.w ) {                               opacity = op1.w;                  }
	else if( z >= z0+di.z ) { float a = (z-(z0+di.z))/di.w; opacity = mix( op1.z, op1.w, a ); }
	else if( z >= z0+di.y ) { float a = (z-(z0+di.y))/di.z; opacity = mix( op1.y, op1.z, a ); }
	else if( z >= z0+di.x ) { float a = (z-(z0+di.x))/di.y; opacity = mix( op1.x, op1.y, a ); }
	else if( z >= z0      ) { float a = (z- z0      )/di.x; opacity = mix( 0.0,   op1.x, a ); }

	return opacity;
}

float getOpMRT( vec2 uv, float z, float z0 )
{
	float opacity = 0.0;

	vec4 op1 = vec4( 0.0 );
	vec4 op2 = vec4( 0.0 );

	for( int i=-2; i<=2 ; i++ )
	for( int j=-2; j<=2 ; j++ )
	{{
		vec2 sample;
		sample.x = uv.x - i * ( 1.0 / viewport.x );
		sample.y = uv.y - j * ( 1.0 / viewport.y );
		op1 += texture2D( opacityMap1, sample );
		op2 += texture2D( opacityMap2, sample );
	}}
	op1 /= vec4( 25.0 );
	op2 /= vec4( 25.0 );

	float d0 = (1.0-z0)/45.0;	// 8+1 layers
	vec4  di = vec4 ( d0, 2.0*d0, 3.0*d0, 4.0*d0 );
	vec4  dj = vec4 ( 5.0*d0, 6.0*d0, 7.0*d0, 8.0*d0 );

	if     ( z >= z0+dj.w ) {                               opacity = op2.w;                }
	else if( z >= z0+dj.z ) { float a = (z-(z0+dj.z))/dj.w; opacity = mix(op2.z, op2.w, a); }
	else if( z >= z0+dj.y ) { float a = (z-(z0+dj.y))/dj.z; opacity = mix(op2.y, op2.z, a); }
	else if( z >= z0+dj.x ) { float a = (z-(z0+dj.x))/dj.y; opacity = mix(op2.x, op2.y, a); }
	else if( z >= z0+di.w ) { float a = (z-(z0+di.w))/dj.x; opacity = mix(op1.w, op2.x, a); }
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

	float opacity = 0.0;
	if( useMRT==0 ) { opacity = getOp   ( uv, z, z0 ); }
	else            { opacity = getOpMRT( uv, z, z0 ); }

	//////////////////////
	// Kajiya-Kay model //

	vec3 L = normalize( lightDir );
	vec3 V = normalize( eyeDir   );
	vec3 T = normalize( hairDir  );

	/////////////
	// ambient //

	vec3 H = gl_LightSource[0].ambient * gl_Color + gl_LightModel.ambient * gl_Color;

	/////////////
	// diffuse //

	float dotTL = dot( T, L );
	float sinTL = sqrt( 1.0 - dotTL*dotTL );
	H += gl_LightSource[0].diffuse * gl_Color * sinTL;

	//////////////
	// specular //

	float dotTV = dot( T, V );
	float sinTV = sqrt( 1.0 - dotTV*dotTV );
	float spec  = pow( max ((dotTL*dotTV+sinTL*sinTV), 0.0), gl_FrontMaterial.shininess );
	H += gl_LightSource[0].specular * gl_FrontMaterial.specular * spec;

	/////////////////
	// self-shadow //

	if( useSelfshadow == 1 )
	{
		H.xyz *= exp( -transmittance * opacity ); 
	}

	//////////////////
	// final result //

	gl_FragColor = vec4( H.xyz, gl_Color.a );
}

