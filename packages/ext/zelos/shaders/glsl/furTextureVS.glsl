//----------------//
// furTextureVS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

void main() 
{
	gl_Position = gl_Vertex;
	//gl_TexCoord[0] = gl_MultiTexCoord0;
	gl_TexCoord[0] = vec4(0.5)*gl_Vertex + vec4(0.5);
}

