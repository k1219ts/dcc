//------------------//
// furKajiyaKayVS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

uniform mat4 shadowMatrix;  // P_l * M_l (* M_e^{-1}) ( except for bias matrix)
uniform vec3 lightEye;      // gl_LightSource[0].position.xyz
varying vec3 lightDir;
varying vec3 hairDir;
varying vec3 eyeDir;
varying vec4 shadowCoordReadOnly;

void main()
{
	gl_Position         = ftransform();
	vec4 vertex         = gl_ModelViewMatrix * gl_Vertex;
	shadowCoordReadOnly = shadowMatrix * gl_Vertex;
	gl_FrontColor       = gl_Color;
	gl_BackColor        = gl_Color;

	lightDir = normalize( lightEye - vertex.xyz );
	hairDir  = gl_NormalMatrix * gl_Normal;
	eyeDir   = -normalize( vertex.xyz );
}

