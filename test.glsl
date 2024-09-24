
// Hash function for randomness
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// Smooth noise
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
               mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

// Fractal noise with multiple scales
float fractalNoise(vec2 uv, float scale) {
    float n = 0.0;
    float s = 1.0;
    for (int i = 0; i < 4; i++) {
        n += noise(uv * scale * s) / s;
        s *= 2.0;
    }
    return n;
}

// Cosine-based palette function
vec3 pal(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * cos(6.28318 * (c * t + d));
}

// Select a random palette
vec3 getRandomPalette(float t, float seed) {
    int paletteId = int(mod(floor(seed * 1000.0), 1.0));
    if (paletteId == 0) return pal(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.33, 0.67));
    if (paletteId == 1) return pal(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.0, 0.10, 0.20));
    if (paletteId == 2) return pal(t, vec3(0.5), vec3(0.5), vec3(1.0), vec3(0.3, 0.20, 0.20));
    if (paletteId == 3) return pal(t, vec3(0.5), vec3(0.5), vec3(1.0, 1.0, 0.5), vec3(0.8, 0.90, 0.30));
    if (paletteId == 4) return pal(t, vec3(0.5), vec3(0.5), vec3(1.0, 0.7, 0.4), vec3(0.0, 0.15, 0.20));
    if (paletteId == 5) return pal(t, vec3(0.5), vec3(0.5), vec3(2.0, 1.0, 0.0), vec3(0.5, 0.20, 0.25));
    return pal(t, vec3(0.8, 0.5, 0.4), vec3(0.2, 0.4, 0.2), vec3(2.0, 1.0, 1.0), vec3(0.0, 0.25, 0.25));
}

// Randomly transform UV coordinates with scaling, rotation, and translation
vec2 transformUV(vec2 uv, float seed) {
    float randomScale = mix(1.0, 10.0, fract(seed * 1000.0));  // Random scale
    float randomRotation = mix(0.0, 6.28318, fract(seed * 100.0));  // Random rotation
    vec2 randomTranslation = vec2(fract(seed * 200.0), fract(seed * 300.0)) * 2.0 - 1.0;  // Random translation

    // Scale and rotate
    uv *= randomScale;
    float s = sin(randomRotation);
    float c = cos(randomRotation);
    mat2 rot = mat2(c, -s, s, c);
    uv = rot * uv;

    // Apply translation
    uv += randomTranslation;

    return uv;
}

// Probability inclusion function
float patternProbability(float seed) {
    return fract(sin(seed * 1000.0)) > 0.5 ? 1.0 : 0.0;
}

// Checkerboard pattern
float checkerPattern(vec2 uv) {
    vec2 grid = floor(uv);
    return mod(grid.x + grid.y, 2.0);
}

// Concentric circles pattern
float circlesPattern(vec2 uv) {
    float r = length(uv);
    return step(0.5, mod(r, 1.0));
}

// Herringbone pattern
float herringbonePattern(vec2 uv) {
    float zigzag = abs(sin(uv.x + mod(uv.y, 2.0) * 5.0));
    return zigzag * 0.5;
}

// Polka dots pattern
float polkaDotsPattern(vec2 uv) {
    vec2 grid = floor(uv);
    vec2 dotPos = fract(uv);
    return smoothstep(0.2, 0.4, length(dotPos - 0.5));
}

// Triangle pattern
float trianglePattern(vec2 uv) {
    return mod(uv.x + uv.y, 2.0) > 1.0 ? 1.0 : 0.0;
}

// Spiral pattern
float spiralPattern(vec2 uv) {
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    return step(0.5, mod(radius + angle * 10.0, 1.0));
}

// Waves pattern
float wavesPattern(vec2 uv) {
    return sin(uv.y * 10.0 + sin(uv.x * 5.0));
}

// Cross-Hatch pattern
float crossHatchPattern(vec2 uv) {
    float horizontal = abs(sin(uv.y * 10.0));
    float vertical = abs(sin(uv.x * 10.0));
    return horizontal + vertical;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y; // Correct aspect ratio

    // Random seed generation using time
    float time = iTime * 0.3;
    float randomSeed = fract(sin(dot(vec2(time, time), vec2(12.9898, 78.233))) * 43758.5453123);

    // Transform UV coordinates for each pattern with a unique seed
    vec2 uvStripes = transformUV(uv, randomSeed * 1001.0);
    vec2 uvDots = transformUV(uv, randomSeed * 2002.0);
    vec2 uvHexagons = transformUV(uv, randomSeed * 3003.0);
    vec2 uvChecker = transformUV(uv, randomSeed * 4004.0);
    vec2 uvHerringbone = transformUV(uv, randomSeed * 5005.0);
    vec2 uvCircles = transformUV(uv, randomSeed * 6006.0);
    vec2 uvTriangles = transformUV(uv, randomSeed * 7007.0);
    vec2 uvSpirals = transformUV(uv, randomSeed * 8008.0);
    vec2 uvWaves = transformUV(uv, randomSeed * 9009.0);
    vec2 uvCrossHatch = transformUV(uv, randomSeed * 1010.0);

    // Generate patterns with probability of inclusion
    float stripes = patternProbability(randomSeed * 1001.0) * abs(sin(uvStripes.y)) * 0.5;
    float dots = patternProbability(randomSeed * 2002.0) * polkaDotsPattern(uvDots);
    float hexagons = patternProbability(randomSeed * 3003.0) * abs(mod(uvHexagons.x + uvHexagons.y, 0.2) * 10.0);
    float checker = patternProbability(randomSeed * 4004.0) * checkerPattern(uvChecker);
    float herringbone = patternProbability(randomSeed * 5005.0) * herringbonePattern(uvHerringbone);
    float circles = patternProbability(randomSeed * 6006.0) * circlesPattern(uvCircles);
    float triangles = patternProbability(randomSeed * 7007.0) * trianglePattern(uvTriangles);
    float spirals = patternProbability(randomSeed * 8008.0) * spiralPattern(uvSpirals);
    float waves = patternProbability(randomSeed * 9009.0) * wavesPattern(uvWaves);
    float crossHatch = patternProbability(randomSeed * 1010.0) * crossHatchPattern(uvCrossHatch);

    // Combine patterns
    float geomPattern = stripes + dots + hexagons + checker + herringbone + circles + triangles + spirals + waves + crossHatch;

    // Apply fractal noise for texture
    float noiseValue = fractalNoise(uv, 10.0);
    geomPattern += noiseValue * 0.2;

    // Randomly select a color palette
    vec3 color = getRandomPalette(geomPattern + noiseValue, randomSeed);

    // Output final color
    fragColor = vec4(color, 1.0);
}
