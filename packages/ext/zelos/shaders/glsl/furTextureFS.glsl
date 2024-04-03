//----------------//
// furTextureFS.h //
//-------------------------------------------------------//
// author: Junghyun Cho @ Seoul National Univ.           //
//         Wanho Choi @ Dexter Studios                   //
// last update: 2014.05.23                               //
//-------------------------------------------------------//

uniform sampler2D tex;

void main()
{
	gl_FragColor = texture2D( tex, gl_TexCoord[0].xy );
}

