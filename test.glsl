// Hash function for randomness
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

// 2D hash function for Voronoi pattern
vec2 hash2(vec2 p) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
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
    int paletteId = int(mod(floor(seed * 1000.0), 7.0));
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
    return fract(sin(seed * 1000.0)) < 0.1 ? 1.0 : 0.0;
}

// Checkerboard pattern
float checkerPattern(vec2 uv) {
    vec2 grid = floor(uv * 3.0);
    return mod(grid.x + grid.y, 2.0) * 0.9;
}

// Concentric circles pattern
float circlesPattern(vec2 uv) {
    float r = length(uv*10.0);
    // return mod(r, 1.0);
    return step(0.5, mod(r, 1.0)) * 0.9;
}

// Herringbone pattern
float herringbonePattern(vec2 uv) {
    uv *=5.0;
    float diagonal1 = fract((uv.x + uv.y) * 0.5);
    float diagonal2 = fract((uv.x - uv.y) * 0.5);

    // Alternate between diagonal1 and diagonal2
    float pattern = mix(diagonal1, diagonal2, step(0.5, fract(uv.y)));

    // Sharpen the pattern
    pattern = step(0.5, pattern) * 0.9;

    return pattern;
}

// Polka dots pattern
float polkaDotsPattern(vec2 uv) {
    vec2 grid = floor(uv);
    vec2 dotPos = fract(uv);
    return smoothstep(0.2, 0.4, length(dotPos - 0.5));
}

// Triangle pattern
float trianglePattern(vec2 uv) {
    // Scale the UV coordinates
    uv *= 10.0;

    // Create a hexagonal grid
    vec2 r = vec2(1.0, sqrt(3.0));
    vec2 h = r * 0.5;
    vec2 a = mod(uv, r) - h;
    vec2 b = mod(uv - h, r) - h;

    // Determine which triangle in the hexagon we're in
    vec2 gv = dot(a, a) < dot(b, b) ? a : b;

    // Calculate distance to the nearest edge
    float d = max(abs(gv.x) * 0.866025 + gv.y * 0.5, -gv.y);
    return d;
}

// Spiral pattern
float spiralPattern(vec2 uv) {
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    float v = mod(mod(radius + angle * 2.0, 3.141592), 1.0);
    return step(0.5, v) * 0.8;
    return v;
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

// Voronoi pattern
float voronoiPattern(vec2 uv) {
    vec2 i_st = floor(uv);
    vec2 f_st = fract(uv);
    float m_dist = 1.;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            vec2 neighbor = vec2(float(x),float(y));
            vec2 point = hash2(i_st + neighbor);
            point = 0.5 + 0.5*sin(vec2(iTime) + 6.2831*point);
            vec2 diff = neighbor + point - f_st;
            float dist = length(diff);
            m_dist = min(m_dist, dist);
        }
    }
    return m_dist;
}

// Mandala pattern
float mandalaPattern(vec2 uv) {
    float r = length(uv);
    float angle = atan(uv.y, uv.x);
    float pattern = sin(angle * 8.0) * sin(r * 10.0);
    return smoothstep(0.0, 0.1, abs(pattern));
}

// Maze pattern
float mazePattern(vec2 uv) {
    float scale = 10.0;
    vec2 grid = floor(uv * scale);
    vec2 f = fract(uv * scale);

    float h1 = hash(grid);
    float h2 = hash(grid + vec2(1.0, 0.0));
    float h3 = hash(grid + vec2(0.0, 1.0));
    float h4 = hash(grid + vec2(1.0, 1.0));

    float horizontalLine = step(0.5, h1) * step(f.y, 0.1) / scale;
    float verticalLine = step(0.5, h2) * step(f.x, 0.1) / scale;
    float topLine = step(0.5, h3) * step(0.9, f.y) / scale;
    float rightLine = step(0.5, h4) * step(0.9, f.x) / scale;

    return max(max(horizontalLine, verticalLine), max(topLine, rightLine));
}

// Honeycomb pattern
float honeycombPattern(vec2 uv) {
    // Adjust scale
    float scale = 10.0;
    uv *= scale;

    // Calculate hexagon grid
    vec2 r = vec2(1.0, sqrt(3.0));
    vec2 h = r * 0.5;
    vec2 a = mod(uv, r) - h;
    vec2 b = mod(uv - h, r) - h;

    // Calculate distance to hexagon edges
    float d = min(dot(a, a), dot(b, b));

    // Adjust edge thickness
    float thickness = 0.05;

    // Create pattern with smooth edges
    return smoothstep(thickness, thickness + 0.1, d);
}

// Fractal pattern
float fractalPattern(vec2 uv) {
    float f = 0.0;
    mat2 m = mat2(1.6, 1.2, -1.2, 1.6);
    f += 0.5000 * noise(uv); uv = m * uv;
    f += 0.2500 * noise(uv); uv = m * uv;
    f += 0.1250 * noise(uv); uv = m * uv;
    f += 0.0625 * noise(uv);
    return f;
}

// Lava lamp pattern
float lavaLampPattern(vec2 uv) {
    float t = iTime * 0.1;
    vec2 n = vec2(sin(t), cos(t));
    float blob1 = length(uv - 0.5 * n);
    float blob2 = length(uv + 0.5 * n);
    return smoothstep(0.4, 0.5, min(blob1, blob2));
}

// Interference pattern
float interferencePattern(vec2 uv) {
    float angle = atan(uv.y, uv.x);
    float radius = length(uv);
    return sin(radius * 20.0 + angle * 10.0 + iTime);
}

// Wood grain pattern
float woodGrainPattern(vec2 uv) {
    float noise = fractalNoise(uv * 10.0, 1.0);
    return fract(noise * 10.0 + uv.x * 2.0);
}

// Brick pattern
float brickPattern(vec2 uv) {
    vec2 pos = vec2(uv.x * 10.0, uv.y * 5.0);
    pos.x += step(1.0, mod(pos.y, 2.0)) * 0.5; // offset every other row
    vec2 sq = fract(pos);
    float brick = step(0.1, sq.x) * step(0.1, sq.y);
    return 1.0 - brick * 0.5;
}

// Perlin warp pattern
float perlinWarpPattern(vec2 uv) {
    vec2 warp = vec2(
        fractalNoise(uv + iTime * 0.1, 4.0),
        fractalNoise(uv + vec2(2.3, 1.7) + iTime * 0.1, 4.0)
    );
    return fractalNoise(uv + warp * 0.5, 4.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy;
    uv -= 0.5;
    uv.x *= iResolution.x / iResolution.y; // Correct aspect ratio

    // Random seed generation using time
    float time = floor(iTime) * 0.3;
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
    vec2 uvVoronoi = transformUV(uv, randomSeed * 1111.0);
    vec2 uvMandala = transformUV(uv, randomSeed * 1212.0);
    vec2 uvMaze = transformUV(uv, randomSeed * 1313.0);
    vec2 uvHoneycomb = transformUV(uv, randomSeed * 1414.0);
    vec2 uvFractal = transformUV(uv, randomSeed * 1515.0);
    vec2 uvLavaLamp = transformUV(uv, randomSeed * 1616.0);
    vec2 uvInterference = transformUV(uv, randomSeed * 1717.0);
    vec2 uvWoodGrain = transformUV(uv, randomSeed * 1818.0);
    vec2 uvBrick = transformUV(uv, randomSeed * 1919.0);
    vec2 uvPerlinWarp = transformUV(uv, randomSeed * 2020.0);

    // Generate patterns with probability of inclusion
    float stripes = abs(sin(uvStripes.y)) * 0.5;
    float dots = polkaDotsPattern(uvDots);
    float hexagons = abs(mod(uvHexagons.x + uvHexagons.y, 0.2) * 10.0);
    float checker = checkerPattern(uvChecker);
    float herringbone = herringbonePattern(uvHerringbone);
    float circles = circlesPattern(uvCircles);
    float triangles = trianglePattern(uvTriangles);
    float spirals = spiralPattern(uvSpirals);
    float waves = wavesPattern(uvWaves);
    float crossHatch = crossHatchPattern(uvCrossHatch);
    float voronoi = voronoiPattern(uvVoronoi);
    float mandala = mandalaPattern(uvMandala);
    float maze = mazePattern(uvMaze);
    float honeycomb = honeycombPattern(uvHoneycomb);
    float fractal = fractalPattern(uvFractal);
    float lavaLamp = lavaLampPattern(uvLavaLamp);
    float interference = interferencePattern(uvInterference);
    float woodGrain = woodGrainPattern(uvWoodGrain);
    float brick = brickPattern(uvBrick);
    float perlinWarp = perlinWarpPattern(uvPerlinWarp);

    // Combine patterns
float geomPattern =
        patternProbability(randomSeed * 2002.0) * polkaDotsPattern(uvDots) +
        patternProbability(randomSeed * 4004.0) * checkerPattern(uvChecker) +
        patternProbability(randomSeed * 5005.0) * herringbonePattern(uvHerringbone) +
        patternProbability(randomSeed * 6006.0) * circlesPattern(uvCircles) +
        patternProbability(randomSeed * 7007.0) * trianglePattern(uvTriangles) +
        patternProbability(randomSeed * 8008.0) * spiralPattern(uvSpirals) +
        patternProbability(randomSeed * 9009.0) * wavesPattern(uvWaves) +
        patternProbability(randomSeed * 1010.0) * crossHatchPattern(uvCrossHatch) +
        patternProbability(randomSeed * 1111.0) * voronoiPattern(uvVoronoi) +
        patternProbability(randomSeed * 1212.0) * mandalaPattern(uvMandala) +
        patternProbability(randomSeed * 1313.0) * mazePattern(uvMaze) +
        patternProbability(randomSeed * 1414.0) * honeycombPattern(uvHoneycomb) +
        patternProbability(randomSeed * 1515.0) * fractalPattern(uvFractal * 10.0) +
        patternProbability(randomSeed * 1616.0) * lavaLampPattern(uvLavaLamp) +
        patternProbability(randomSeed * 1717.0) * interferencePattern(uvInterference) +
        patternProbability(randomSeed * 1818.0) * woodGrainPattern(uvWoodGrain/5.0) +
        patternProbability(randomSeed * 1919.0) * brickPattern(uvBrick) +
        patternProbability(randomSeed * 2020.0) * perlinWarpPattern(uvPerlinWarp) +
        // /* patternProbability(randomSeed * 1001.0) * */ abs(sin(uvStripes.y)) * 0.5 +
        // /* patternProbability(randomSeed * 3003.0) * */ abs(mod(uvHexagons.x + uvHexagons.y, 0.2) * 10.0) +
        0;

    // Apply fractal noise for texture
    // float noiseValue = fractalNoise(uv, 10.0);
    // geomPattern += noiseValue * 0.2;

    // Randomly select a color palette
    vec3 color = getRandomPalette(geomPattern, randomSeed);

    // Output final color
    fragColor = vec4(color, 1.0);
}
