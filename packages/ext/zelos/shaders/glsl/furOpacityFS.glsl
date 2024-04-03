//----------------//
// furOpacityFS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

// Render a depth map of the hair as seen from the light source
// For each pixel, the first depth z0 is given.
// z0, z0+d1, z0+d2, z0+d3, d_{k-1} < d_{k}

uniform sampler2D depthMap;
uniform vec2      viewport;
uniform float     opacity;
uniform int       useMRT;

void main()
{
	vec2 texCoord = vec2( gl_FragCoord.x / viewport.x, gl_FragCoord.y / viewport.y );
	float z  = gl_FragCoord.z;
	float z0 = texture2D( depthMap, texCoord ).x; // DEPTH_TEXTURE_MODE: LUMINANCE = rrr1
	float op = opacity;

	if( useMRT == 0 ) {

		float d0 = (1.0-z0)/15.0;   // 4+1 layers
		vec4  di = vec4( d0, 2.0*d0, 3.0*d0, 4.0*d0 );
		if     ( z >= z0+di.w ) { gl_FragData[0] = vec4( 0.0, 0.0, 0.0, op ); }
		else if( z >= z0+di.z ) { gl_FragData[0] = vec4( 0.0, 0.0, 0.0, op ); }
		else if( z >= z0+di.y ) { gl_FragData[0] = vec4( 0.0, 0.0, op,  op ); }
		else if( z >= z0+di.x ) { gl_FragData[0] = vec4( 0.0, op,  op,  op ); }
		else                    { gl_FragData[0] = vec4( op ); }
		// The 1st layer(R) will be accumulated less
		// while the 3rd layer(B) will be accumulated more

	} else {

		float d0 = (1.0-z0)/45.0;	// 8+1 layers
		vec4  di = vec4( d0, 2.0*d0, 3.0*d0, 4.0*d0 );
		vec4  dj = vec4( 5.0*d0, 6.0*d0, 7.0*d0, 8.0*d0 );
		if     ( z >= z0+dj.w ) { gl_FragData[0] = vec4( 0.0 );               gl_FragData[1] = vec4( 0.0, 0.0, 0.0, op ); }
		else if( z >= z0+dj.z ) { gl_FragData[0] = vec4( 0.0 );               gl_FragData[1] = vec4( 0.0, 0.0, 0.0, op ); }
		else if( z >= z0+dj.y ) { gl_FragData[0] = vec4( 0.0 );               gl_FragData[1] = vec4( 0.0, 0.0, op,  op ); }
		else if( z >= z0+dj.x ) { gl_FragData[0] = vec4( 0.0 );               gl_FragData[1] = vec4( 0.0, op,  op,  op ); }
		else if( z >= z0+di.w ) { gl_FragData[0] = vec4( 0.0 );               gl_FragData[1] = vec4( op ); }
		else if( z >= z0+di.z ) { gl_FragData[0] = vec4( 0.0, 0.0, 0.0, op ); gl_FragData[1] = vec4( op ); }
		else if( z >= z0+di.y ) { gl_FragData[0] = vec4( 0.0, 0.0, op, op );  gl_FragData[1] = vec4( op ); }
		else if( z >= z0+di.x ) { gl_FragData[0] = vec4( 0.0, op, op, op );   gl_FragData[1] = vec4( op ); }
		else                    { gl_FragData[0] = vec4( op );                gl_FragData[1] = vec4( op ); }

	}
}

