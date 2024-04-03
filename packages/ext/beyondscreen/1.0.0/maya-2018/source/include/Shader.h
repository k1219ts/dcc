#ifndef _BS_ScreenMeshShader_h_
#define _BS_ScreenMeshShader_h_

#include <BeyondScreen.h>

///////////////////
// Vertex Shader //
static const char* ScreenMeshVS =
R"(
#version 430 core

// uniform input data
uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;

// per vertex input data
layout(location=0) in vec4 vPOS;
layout(location=1) in vec4 vUVW;

// output to the fragment shader
out vec3 fPOS;

void main()
{
	//gl_Position = projectionMatrix * modelViewMatrix * vPOS;
    //fPOS = vPOS.xyz;

	//gl_Position = vUVW;
	gl_Position = projectionMatrix * modelViewMatrix * vUVW;
    fPOS = vPOS.xyz;
}

)";

/////////////////////
// Fragment Shader //
static const char* ScreenMeshFS =
R"(
#version 430 core

// uniform input data
uniform vec3      worldAimingPoint;
uniform vec3      worldCameraPosition;
uniform vec3      worldCameraUpvector;
uniform int       hasFisheyeTexture;
uniform sampler2D fisheyeTexture;

// per fragment input data
layout(location=0) in vec3 fPOS;

out vec4 outColor;

void main()
{
    vec3 zAxis = normalize( worldAimingPoint - worldCameraPosition );
    vec3 xAxis = normalize( cross( zAxis, worldCameraUpvector ) );
    vec3 yAxis = normalize( cross( xAxis, zAxis ) );

    vec3 direction = normalize( fPOS - worldCameraPosition );

    float xValue = dot( direction, xAxis );
    float yValue = dot( direction, yAxis );
    float zValue = dot( direction, zAxis );

    vec3 projectedDirection = ( xValue * xAxis ) + ( yValue * yAxis );

    float alpha = 1.0 - acos( length( projectedDirection ) ) / ( 0.5 * 3.1415926535897931159979634685441851615906 );

    projectedDirection = normalize( projectedDirection );
    projectedDirection *= alpha;

    float x = dot( projectedDirection, xAxis );
    float y = dot( projectedDirection, yAxis );

    float s = ( 0.5 * x ) + 0.5;
    float t = ( 0.5 * y ) + 0.5;

    outColor = texture2D( fisheyeTexture, vec2(s,-t) );
}

)";

#endif

