#version 330 core

in vec2 TextureCoordinate;

uniform sampler2D ColorTexture;
uniform sampler2D BrightnessTexture;

layout(location = 0) out vec4 out_Color;

void main(void) {

	/* vec4 color = texture2D(ColorTexture, TextureCoordinate);
	
	float red = color.r;
	float green = color.g;
	float blue = color.b;	
	
	float minimumRGB = min(red ,min(green, blue));
	float maximumRGB = max(red ,max(green, blue));
	
	float minimumPlusMaximum = minimumRGB + maximumRGB;
	
	out_Color = vec4(minimumPlusMaximum - red, minimumPlusMaximum - green, minimumPlusMaximum - blue, 1.0f); */
	
	vec4 color = vec4(0.0f);

	for(int i= -4 ;i < 4; i++)
		for(int j = -3; j < 3; j++)
			color += texture2D(ColorTexture, TextureCoordinate + vec2(j, i) * 0.004) * 0.25;
		
	if (texture2D(ColorTexture, TextureCoordinate).r < 0.3) {

		out_Color = color * color * 0.012 + texture2D(ColorTexture, TextureCoordinate);
	}
	else {
	
		if (texture2D(ColorTexture, TextureCoordinate).r < 0.5) {
		
			out_Color = color * color * 0.009 + texture2D(ColorTexture, TextureCoordinate);
		}
		else {
		
			out_Color = color * color * 0.0075 + texture2D(ColorTexture, TextureCoordinate);
		}
	}	
}