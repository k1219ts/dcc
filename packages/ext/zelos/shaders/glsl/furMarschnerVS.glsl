//------------------//
// furMarschnerVS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

uniform vec3  lightWorld;    // The world position of light
uniform vec3  eyeWorld;      // The world position of eye
uniform mat4  shadowMatrix;  // P_l * M_l (* M_e^{-1}) ( except for bias matrix)
varying vec4  shadowCoordReadOnly;
varying float sin_theta_r;
varying float sin_theta_i;
varying float cos_phi;
varying float cos_theta_i;
varying float cos_phi_H;

void main() 
{
	gl_Position         = ftransform();
	shadowCoordReadOnly = shadowMatrix * gl_Vertex;
	gl_FrontColor       = gl_Color;
	gl_BackColor        = gl_Color;

	vec3 L = normalize( lightWorld - gl_Vertex.xyz );
	vec3 V = normalize( eyeWorld - gl_Vertex.xyz );
	vec3 T = gl_Normal;

	sin_theta_i = dot( L, T );
	sin_theta_r = dot( V, T );

	// angle between L and V in XY-plane
	vec3 lightPerp = L - ( sin_theta_i * T );
	vec3 eyePerp   = V - ( sin_theta_r * T );
	float l_lightPerp = dot( lightPerp, lightPerp );
	float l_eyePerp   = dot( eyePerp,   eyePerp   );
	cos_phi = dot( eyePerp, lightPerp ) * pow( l_lightPerp*l_eyePerp, -0.5 );

	// We need these for our two axis diffuse model
	cos_theta_i = sqrt( 1.0 - sin_theta_i * sin_theta_i );
	cos_phi_H   = cos( acos(cos_phi) * 0.5 );
}

