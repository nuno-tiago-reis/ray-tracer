#version 330 core

in vec2 TextureCoordinate;

uniform sampler2D Texture0;
uniform sampler2D Texture1;

layout(location = 0) out vec4 out_Color;

void main(void) {

	out_Color = texture2D(Texture0, TextureCoordinate) * 0.57;

	out_Color += texture2D(Texture1, TextureCoordinate) * 0.43;
}