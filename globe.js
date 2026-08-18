"use strict";
/* globe.js: a from-scratch WebGL2 globe renderer for trajsim.
 *
 * Why it exists: the app's physics lives in ECEF on a sphere of radius
 * R_EARTH, and MapLibre's flat-world camera model fought that all evening
 * (no free camera, zoom/centre coupling, stale-elevation inheritance,
 * flat-plane datum, DOM mirror-projection). This renderer puts the graphics
 * in the same frame as the physics: the camera is a position and an
 * orientation in ECEF, tiles are meshes on the real sphere displaced by real
 * terrain heights, and everything drawn takes [lat, lon, height above MSL].
 *
 * Sources: OSM raster imagery (browser-direct, CORS open), Mapzen/AWS
 * terrarium elevation tiles via the local server proxy (/tile/terrarium,
 * the bucket sends no CORS headers), OpenFreeMap vector tiles for the
 * building layer (phase 2).
 *
 * Precision scheme: all anchors and the camera are computed in JS doubles;
 * the GPU only ever sees positions relative to a nearby anchor (float32 is
 * exact to well under a meter at those magnitudes). Depth is logarithmic,
 * written in the vertex shader, identical formula in every program, so a
 * 2 m launch pin and a horizon 3,000 km away share one depth buffer.
 */

const GLOBE_R = 6371000;               // must match the app's R_EARTH
const FOVY = 0.6435011087932844;       // rad, MapLibre's default: keeps the
                                       // fly-cam feel the ramps were tuned to
const MAX_Z = 17;                      // imagery depth cap (as before)
const TERRAIN_Z = 12;                  // terrarium sampling zoom (~38 m/px)
const BASE_Z = 2;                      // always-resident world coverage
const REFINE_PX = 384;                 // split a tile when it covers more px
const TEX_CAP = 700;                   // LRU imagery textures
const HGT_CAP = 320;                   // LRU terrain grids
const MAX_IMG_INFLIGHT = 10;
const MAX_HGT_INFLIGHT = 6;
const HGT_RETRY_MS = 1500;    // backoff before refetching a failed tile
const ATMO_EXP = 24.0;        // HDR->display exposure for the physical sky
const LOG_FAR = 1e8;                   // log-depth far plane, m
const MAXLAT = 85.05112877980659;      // web-mercator latitude limit

/* ---------- small math ---------- */
const D2R = Math.PI / 180, R2D = 180 / Math.PI;
function ecef(lat, lon, h) {
  const la = lat * D2R, lo = lon * D2R, r = GLOBE_R + h;
  const c = Math.cos(la);
  return [r * c * Math.cos(lo), r * c * Math.sin(lo), r * Math.sin(la)];
}
function vsub(a, b) { return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }
function vdot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function vcross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function vnorm(a) { return Math.hypot(a[0], a[1], a[2]); }
function vunit(a) { const n = vnorm(a) || 1; return [a[0]/n, a[1]/n, a[2]/n]; }

/* ---------- terrarium PNG -> heights, without a canvas ----------
 * Elevation arrives as a PNG whose RGB IS a number (h = R*256 + G + B/256
 * - 32768), so it must be decoded exactly. A canvas decode is not exact:
 * Brave's fingerprint shield perturbs readbacks by +-1 per channel, and
 * +-1 in the red channel is +-256 m, which painted deterministic pyramids
 * into the terrain (see HANDOFF). Inflate comes from the platform's
 * DecompressionStream; the rest is the PNG spec's five scanline filters.
 * A buffer that is not a PNG is assumed to be raw float32 heights (the
 * local dev server's .bin route). */
async function decodeTerrarium(buf) {
  const u8 = new Uint8Array(buf);
  if (!(u8[0] === 0x89 && u8[1] === 0x50 && u8[2] === 0x4e && u8[3] === 0x47))
    return new Float32Array(buf);
  const dv = new DataView(buf);
  let pos = 8, w = 0, h = 0, depth = 0, ctype = 0, interlace = 0;
  const idat = [];
  while (pos + 8 <= u8.length) {
    const len = dv.getUint32(pos);
    const type = String.fromCharCode(u8[pos+4], u8[pos+5], u8[pos+6], u8[pos+7]);
    const body = pos + 8;
    if (type === "IHDR") {
      w = dv.getUint32(body); h = dv.getUint32(body + 4);
      depth = u8[body + 8]; ctype = u8[body + 9]; interlace = u8[body + 12];
    } else if (type === "IDAT") idat.push(u8.subarray(body, body + len));
    else if (type === "IEND") break;
    pos = body + len + 4;                       // + CRC
  }
  const chan = ctype === 2 ? 3 : ctype === 6 ? 4 : ctype === 0 ? 1 : 0;
  if (depth !== 8 || interlace !== 0 || !chan)
    throw new Error(`unsupported PNG: depth ${depth} type ${ctype} interlace ${interlace}`);
  let total = 0;
  for (const c of idat) total += c.length;
  const z = new Uint8Array(total);
  let o = 0;
  for (const c of idat) { z.set(c, o); o += c.length; }
  const raw = new Uint8Array(await new Response(
    new Blob([z]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
  const stride = w * chan, px = new Uint8Array(stride * h);
  let p = 0;
  for (let y = 0; y < h; y++) {
    const f = raw[p++], row = y * stride, prev = row - stride;
    for (let x = 0; x < stride; x++) {
      const cur = raw[p + x];
      const a = x >= chan ? px[row + x - chan] : 0;
      const b = y > 0 ? px[prev + x] : 0;
      const c = (x >= chan && y > 0) ? px[prev + x - chan] : 0;
      let v;
      if (f === 0) v = cur;
      else if (f === 1) v = cur + a;
      else if (f === 2) v = cur + b;
      else if (f === 3) v = cur + ((a + b) >> 1);
      else {                                     // 4: Paeth
        const q = a + b - c;
        const pa = Math.abs(q - a), pb = Math.abs(q - b), pc = Math.abs(q - c);
        v = cur + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
      }
      px[row + x] = v & 255;
    }
    p += stride;
  }
  const g = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const j = i * chan;
    g[i] = chan === 1 ? px[j] * 256 - 32768
                      : px[j] * 256 + px[j + 1] + px[j + 2] / 256 - 32768;
  }
  return g;
}

/* tile <-> geo (web mercator) */
// normalized: a camera that keeps flying west accumulates longitude past
// -180 (destPoint just adds), and an un-wrapped value here yields negative
// tile indices, so terrain lookups silently miss and fetches 404
function lon2tx(lon, z) { return (((lon + 180) % 360 + 360) % 360) / 360 * (1 << z); }
function lat2ty(lat, z) {
  const la = Math.max(-MAXLAT, Math.min(MAXLAT, lat)) * D2R;
  return (1 - Math.asinh(Math.tan(la)) / Math.PI) / 2 * (1 << z);
}
function ty2lat(y, z) { return Math.atan(Math.sinh(Math.PI * (1 - 2 * y / (1 << z)))) * R2D; }
function tx2lon(x, z) { return x / (1 << z) * 360 - 180; }
const tkey = (z, x, y) => z + "/" + x + "/" + y;

/* ---------- shaders ---------- */
const ATMO_GLSL = `
const float Rg = 6371000.0;
const float Rt = 6471000.0;
const vec3  betaR  = vec3(5.802e-6, 13.558e-6, 33.1e-6);
const float betaMs = 3.996e-6;
const float betaMe = 4.40e-6 + 3.996e-6;
const vec3  betaO  = vec3(0.650e-6, 1.881e-6, 0.085e-6);
float dR(float h) { return exp(-h / 8000.0); }
float dM(float h) { return exp(-h / 1200.0); }
float dO(float h) { return max(0.0, 1.0 - abs(h - 25000.0) / 15000.0); }
vec3 extinct(float h) { return betaR * dR(h) + vec3(betaMe * dM(h)) + betaO * dO(h); }
float ssqrt(float x) { return sqrt(max(x, 0.0)); }
float distToTop(float r, float mu) {
  return max(0.0, -r * mu + ssqrt(r * r * (mu * mu - 1.0) + Rt * Rt));
}
float distToGround(float r, float mu) {
  float disc = r * r * (mu * mu - 1.0) + Rg * Rg;
  if (disc < 0.0) return -1.0;
  float d = -r * mu - ssqrt(disc);
  return d >= 0.0 ? d : -1.0;
}
vec2 ttUv(float r, float mu) {
  r = clamp(r, Rg + 1.0, Rt - 1.0);
  float H = ssqrt(Rt * Rt - Rg * Rg);
  float rho = ssqrt(r * r - Rg * Rg);
  float d = distToTop(r, mu);
  float dMin = Rt - r, dMax = rho + H;
  return vec2((d - dMin) / max(1.0, dMax - dMin), rho / H);
}
// phase functions
float phR(float c) { return 3.0 / (16.0 * 3.14159265) * (1.0 + c * c); }
float phM(float c) {
  float g = 0.8, g2 = 0.64;
  return 3.0 / (8.0 * 3.14159265) * (1.0 - g2) * (1.0 + c * c)
       / ((2.0 + g2) * pow(1.0 + g2 - 2.0 * g * c, 1.5));
}
// sky-view LUT warp: sqrt-spaced around the horizon of radius r
vec2 svUv(float r, float theta, float phi) {
  float chor = -ssqrt(r * r - Rg * Rg) / r;
  float hor = acos(clamp(chor, -1.0, 1.0));
  float v = theta < hor
    ? 0.5 * (1.0 - sqrt(max(0.0, 1.0 - theta / hor)))
    : 0.5 + 0.5 * sqrt(max(0.0, (theta - hor) / max(1e-4, 3.14159265 - hor)));
  return vec2(phi / 6.2831853 + 0.5, v);
}
`;

const TILE_VS = `#version 300 es
precision highp float;
in vec3 aPos; in vec2 aUV;
uniform mat4 uVP; uniform vec3 uOrigin; uniform float uLogF;
uniform vec4 uUVT;            // uv transform: scale, scale, offx, offy
out vec2 vUV; out float vDist; out vec3 vPos;
void main() {
  vec3 p = aPos + uOrigin;
  vec4 pos = uVP * vec4(p, 1.0);
  vDist = pos.w;
  ${"pos.z = (log2(max(1e-6, pos.w + 1.0)) * uLogF - 1.0) * pos.w;"}
  gl_Position = pos;
  vUV = aUV * uUVT.xy + uUVT.zw;
  vPos = p;
}`;

const TILE_FS = `#version 300 es
precision highp float;   // mediump = fp16 on some GPUs; vPos (~6.4e6 m) and
                         // uAerMax overflow it, NaN-ing the day/night factor
in vec2 vUV; in float vDist; in vec3 vPos;
uniform sampler2D uTex;
uniform vec4 uSolid;          // a > 0: flat color (polar caps), no texture
uniform vec3 uFogCol; uniform float uFogNear; uniform float uFogFar;
uniform int uMode;            // 0 analytic fog, 1 physical aerial perspective
uniform vec3 uSun; uniform vec3 uEarthC; uniform float uExp;
uniform vec3 uUpW; uniform vec3 uAzX; uniform vec3 uAzY; uniform float uCamR;
uniform sampler2D uSkyV;
out vec4 frag;
${ATMO_GLSL}
// mean density along the chord camera->fragment by composite Simpson on
// the exact geometric heights (5 points, 2 segments); closed-form, no LUT
vec3 tauChord(vec3 a, vec3 b) {
  vec3 t = vec3(0.0);
  vec3 pts[5];
  pts[0] = a; pts[4] = b;
  pts[2] = mix(a, b, 0.5); pts[1] = mix(a, b, 0.25); pts[3] = mix(a, b, 0.75);
  float hR = 0.0, hM = 0.0, hO = 0.0;
  float wgt[5] = float[5](1.0, 4.0, 2.0, 4.0, 1.0);
  for (int i = 0; i < 5; i++) {
    float h = max(0.0, length(pts[i]) - Rg);
    hR += wgt[i] * dR(h); hM += wgt[i] * dM(h); hO += wgt[i] * dO(h);
  }
  float L = distance(a, b) / 12.0;
  return betaR * (hR * L) + vec3(betaMe * hM * L) + betaO * (hO * L);
}
void main() {
  vec3 col = uSolid.a > 0.0 ? uSolid.rgb : texture(uTex, vUV).rgb;
  if (uMode == 1) {
    // day/night: dim the daylit raster by sun elevation at the surface
    vec3 sdir = normalize(vPos - uEarthC);
    float dim = 0.55 + 0.45 * smoothstep(-0.10, 0.12, dot(sdir, uSun));
    col *= dim;
    // aerial perspective from PROVEN ingredients only (the froxel volume
    // misrendered on real GPUs): in-scatter color comes from the sky-view
    // LUT along this pixel's direction (whose below-horizon rows already
    // hold path-to-ground in-scatter), scaled by how much of the full
    // path's opacity this fragment's distance covers; opacity itself is a
    // closed-form Simpson integral of the density profile
    float dist = length(vPos);
    vec3 rayd = vPos / dist;
    float theta = acos(clamp(dot(rayd, uUpW), -1.0, 1.0));
    float phi = atan(dot(rayd, uAzY), dot(rayd, uAzX));
    vec4 sv = texture(uSkyV, svUv(uCamR, theta, phi));
    vec3 Tpart = exp(-tauChord(-uEarthC, vPos - uEarthC));
    float frac = clamp((1.0 - Tpart.g) / max(1e-4, 1.0 - sv.a), 0.0, 1.0);
    vec3 inscatter = (vec3(1.0) - exp(-sv.rgb * uExp * 0.26)) * frac;
    // cartographic compositing: soften physical dimming to match the
    // reduced-exposure brightening, or bright map albedo develops a dark
    // mid-oblique saddle (the map is display-space, not radiometric)
    col = col * (1.0 - (1.0 - Tpart.g) * 0.20) + inscatter;
  } else {
    float f = smoothstep(uFogNear, uFogFar, vDist);
    col = mix(col, uFogCol, f * 0.92);
  }
  frag = vec4(col, 1.0);
}`;

const SKY_VS = `#version 300 es
precision highp float;
in vec2 aNdc; out vec2 vNdc;
void main() { vNdc = aNdc; gl_Position = vec4(aNdc, 0.9999, 1.0); }`;

// Analytic clear-sky: color from the ray's closest-approach altitude to the
// sphere. Overhead from the ground that altitude is the camera's own (dense
// air, blue); at the horizon it grazes sea level (white haze band); from
// orbit most rays never touch air (black space) except the thin limb, which
// falls out of the same expression as a blue ring for free.
const SKY_FS = `#version 300 es
precision highp float;
in vec2 vNdc; out vec4 frag;
uniform vec3 uFwd; uniform vec3 uRight; uniform vec3 uUp;
uniform float uTanHalf; uniform float uAspect;
uniform vec3 uEarthC;         // earth center relative to camera, world frame
uniform float uR;
void main() {
  vec3 dir = normalize(uFwd + vNdc.x * uTanHalf * uAspect * uRight
                            + vNdc.y * uTanHalf * uUp);
  float tca = dot(dir, uEarthC);
  float c2 = dot(uEarthC, uEarthC) - tca * tca;
  float camAlt = length(uEarthC) - uR;
  float hmin;
  if (tca > 0.0) {
    float c = sqrt(max(c2, 0.0));
    hmin = c - uR;               // may be negative: ray hits the earth
  } else hmin = camAlt;          // looking away: thinnest air is right here
  float h = max(hmin, 0.0);
  float rho = exp(-h / 8500.0);         // bulk atmosphere
  float graze = exp(-h / 1100.0);       // bright band hugging the horizon
  vec3 space = vec3(0.004, 0.006, 0.012);
  vec3 blue  = vec3(0.345, 0.585, 0.855);
  vec3 haze  = vec3(0.905, 0.935, 0.965);
  vec3 col = mix(space, blue, rho);
  col = mix(col, haze, graze * 0.72);
  if (tca > 0.0 && c2 < uR * uR) col = haze;   // below-horizon backstop
  frag = vec4(col, 1.0);
}`;

// Real stars: the Yale Bright Star Catalog (stars.bin, 9,096 stars, all of
// naked-eye astronomy) as point sprites. Directions are J2000 equatorial
// unit vectors rotated into ECEF by Greenwich sidereal time each frame
// (precession since 2000 is ~0.4 deg, under the naked-eye pattern
// threshold, ignored). Visibility reuses the sky shader's closest-approach
// altitude, so stars fade in exactly where its air thins to black and cut
// off below the horizon; terrain overdraws them afterwards.
const STAR_VS = `#version 300 es
precision highp float;
in vec3 aDir; in float aMag; in float aBv;
uniform mat4 uVP; uniform float uGmst; uniform float uDist;
uniform vec3 uEarthC; uniform float uR; uniform float uPxScale;
uniform float uNight;         // 1 = night sky at ground (physical sky mode)
out float vA; out vec3 vCol;
void main() {
  float c = cos(uGmst), s = sin(uGmst);
  vec3 d = vec3(c*aDir.x + s*aDir.y, -s*aDir.x + c*aDir.y, aDir.z);
  gl_Position = uVP * vec4(d * uDist, 1.0);
  float tca = dot(d, uEarthC);
  float c2 = dot(uEarthC, uEarthC) - tca * tca;
  float camAlt = length(uEarthC) - uR;
  float hmin = tca > 0.0 ? sqrt(max(c2, 0.0)) - uR : camAlt;
  bool hit = tca > 0.0 && c2 < uR * uR;
  float h = max(hmin, 0.0);
  float sky = min(1.0, exp(-h / 8500.0) + 0.72 * exp(-h / 1100.0));
  float vis = hit ? 0.0 : max(1.0 - sky, uNight);
  // zero point 3.0: brighter than strict photometry by ~1.5 mag, the usual
  // planetarium compensation for a screen's inability to render scotopic
  // vision; a mag-5 star is one dim pixel, Sirius saturates
  float flux = pow(10.0, -0.4 * (aMag - 3.0));
  vA = min(1.0, flux) * vis;
  gl_PointSize = uPxScale * clamp(3.0 - 0.30 * aMag, 1.5, 6.0);
  // realistic star colors: blue-white through white (solar B-V ~0.65)
  // to pale orange; nothing in the night sky is actually red
  float t = clamp((aBv + 0.3) / 1.9, 0.0, 1.0);
  vCol = t < 0.5 ? mix(vec3(0.64, 0.76, 1.0), vec3(1.0, 0.99, 0.95), t * 2.0)
                 : mix(vec3(1.0, 0.99, 0.95), vec3(1.0, 0.80, 0.62), t * 2.0 - 1.0);
}`;

const STAR_FS = `#version 300 es
precision highp float;
in float vA; in vec3 vCol; out vec4 frag;
void main() {
  vec2 q = gl_PointCoord * 2.0 - 1.0;
  float r2 = dot(q, q);
  if (r2 > 1.0) discard;
  frag = vec4(vCol * vA * exp(-2.8 * r2), 1.0);
}`;

// screen-space expanded polylines (same scheme the MapLibre custom layer
// used): each segment is a quad, pushed sideways in pixels in the vertex
// shader; vertices behind the camera collapse the primitive
const LINE_VS = `#version 300 es
precision highp float;
in vec3 aA; in vec3 aB; in vec2 aSE;
uniform mat4 uVP; uniform vec3 uOrigin; uniform vec2 uVp;
uniform float uW; uniform float uLogF;
void main() {
  vec4 ca = uVP * vec4(aA + uOrigin, 1.0);
  vec4 cb = uVP * vec4(aB + uOrigin, 1.0);
  if (ca.w < 0.001 || cb.w < 0.001) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
  vec2 pa = ca.xy / ca.w * uVp;
  vec2 pb = cb.xy / cb.w * uVp;
  vec2 d = pb - pa;
  d = length(d) < 1e-6 ? vec2(1.0, 0.0) : normalize(d);
  vec2 nrm = vec2(-d.y, d.x);
  vec4 pos = aSE.y > 0.5 ? cb : ca;
  pos.xy += nrm * aSE.x * uW / uVp * pos.w;
  ${"pos.z = (log2(max(1e-6, pos.w + 1.0)) * uLogF - 1.0) * pos.w;"}
  gl_Position = pos;
}`;

const LINE_FS = `#version 300 es
precision mediump float;
uniform vec4 uC; out vec4 frag;
void main() { frag = vec4(uC.rgb * uC.a, uC.a); }`;

// buildings: flat-shaded extrusions, lambert on the face normal
const BLDG_VS = `#version 300 es
precision highp float;
in vec3 aPos; in vec3 aNrm;
uniform mat4 uVP; uniform vec3 uOrigin; uniform float uLogF;
out vec3 vNrm; out float vDist; out vec3 vPos;
void main() {
  vec3 p = aPos + uOrigin;
  vec4 pos = uVP * vec4(p, 1.0);
  vDist = pos.w;
  ${"pos.z = (log2(max(1e-6, pos.w + 1.0)) * uLogF - 1.0) * pos.w;"}
  gl_Position = pos;
  vNrm = aNrm;
  vPos = p;
}`;

const BLDG_FS = `#version 300 es
precision highp float;   // same fp16-overflow hazard as TILE_FS
in vec3 vNrm; in float vDist; in vec3 vPos;
uniform vec3 uSun; uniform vec3 uFogCol; uniform float uFogNear; uniform float uFogFar;
uniform int uMode; uniform vec3 uEarthC; uniform float uExp;
uniform vec3 uUpW; uniform vec3 uAzX; uniform vec3 uAzY; uniform float uCamR;
uniform sampler2D uSkyV;
out vec4 frag;
${ATMO_GLSL}
vec3 tauChord(vec3 a, vec3 b) {
  vec3 pts[5];
  pts[0] = a; pts[4] = b;
  pts[2] = mix(a, b, 0.5); pts[1] = mix(a, b, 0.25); pts[3] = mix(a, b, 0.75);
  float hR = 0.0, hM = 0.0, hO = 0.0;
  float wgt[5] = float[5](1.0, 4.0, 2.0, 4.0, 1.0);
  for (int i = 0; i < 5; i++) {
    float h = max(0.0, length(pts[i]) - Rg);
    hR += wgt[i] * dR(h); hM += wgt[i] * dM(h); hO += wgt[i] * dO(h);
  }
  float L = distance(a, b) / 12.0;
  return betaR * (hR * L) + vec3(betaMe * hM * L) + betaO * (hO * L);
}
void main() {
  float l = 0.62 + 0.38 * max(dot(normalize(vNrm), uSun), 0.0);
  vec3 col = vec3(0.576, 0.631, 0.702) * l;      // #93a1b3, the old look
  if (uMode == 1) {
    vec3 sdir = normalize(vPos - uEarthC);
    float dim = 0.55 + 0.45 * smoothstep(-0.10, 0.12, dot(sdir, uSun));
    col *= dim;
    float dist = length(vPos);
    vec3 rayd = vPos / dist;
    float theta = acos(clamp(dot(rayd, uUpW), -1.0, 1.0));
    float phi = atan(dot(rayd, uAzY), dot(rayd, uAzX));
    vec4 sv = texture(uSkyV, svUv(uCamR, theta, phi));
    vec3 Tpart = exp(-tauChord(-uEarthC, vPos - uEarthC));
    float frac = clamp((1.0 - Tpart.g) / max(1e-4, 1.0 - sv.a), 0.0, 1.0);
    col = col * (1.0 - (1.0 - Tpart.g) * 0.20)
        + (vec3(1.0) - exp(-sv.rgb * uExp * 0.26)) * frac;
  } else {
    float f = smoothstep(uFogNear, uFogFar, vDist);
    col = mix(col, uFogCol, f * 0.92);
  }
  frag = vec4(col, 1.0);
}`;

/* ---------- physical sky: Hillaire 2020 ----------
   Transmittance + multiple-scattering LUTs precomputed once; a small
   camera-dependent sky-view LUT and a 32x32x16 aerial-perspective froxel
   volume rebuilt each frame; the screen pass is texture lookups. Rayleigh +
   Mie (Cornette-Shanks g=0.8) + ozone, real sun from the clock. Falls back
   to the analytic sky when float render targets are unavailable
   (globe.skyMode = "analytic" forces the fallback). */

const AQUAD_VS = `#version 300 es
precision highp float;
in vec2 aNdc; out vec2 vUV;
void main() { vUV = aNdc * 0.5 + 0.5; gl_Position = vec4(aNdc, 0.0, 1.0); }`;

const TRANS_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
${ATMO_GLSL}
void main() {
  float H = ssqrt(Rt * Rt - Rg * Rg);
  float rho = H * vUV.y;
  float r = ssqrt(rho * rho + Rg * Rg);
  float dMin = Rt - r, dMax = rho + H;
  float d = dMin + vUV.x * (dMax - dMin);
  float mu = d <= 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
  const int N = 44;
  vec3 tau = vec3(0.0);
  float dt = d / float(N);
  for (int i = 0; i < N; i++) {
    float t = (float(i) + 0.5) * dt;
    float rs = ssqrt(r * r + t * t + 2.0 * r * mu * t);
    tau += extinct(rs - Rg) * dt;
  }
  frag = vec4(exp(-tau), 1.0);
}`;

const MULTI_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uT;
${ATMO_GLSL}
void main() {
  float r = mix(Rg + 10.0, Rt - 10.0, vUV.y);
  float muS = vUV.x * 2.0 - 1.0;
  vec3 sun = vec3(ssqrt(1.0 - muS * muS), 0.0, muS);
  vec3 L2 = vec3(0.0), fms = vec3(0.0);
  const int ND = 64;
  for (int i = 0; i < ND; i++) {          // fibonacci sphere directions
    float fi = (float(i) + 0.5) / float(ND);
    float ct = 1.0 - 2.0 * fi;
    float st = ssqrt(1.0 - ct * ct);
    float ph = 2.399963 * float(i);
    vec3 d = vec3(st * cos(ph), st * sin(ph), ct);
    float dg = distToGround(r, d.z);
    float dEnd = dg > 0.0 ? dg : distToTop(r, d.z);
    const int NS = 20;
    float dt = dEnd / float(NS);
    vec3 T = vec3(1.0);
    for (int j = 0; j < NS; j++) {
      float t = (float(j) + 0.5) * dt;
      float rs = ssqrt(r * r + t * t + 2.0 * r * d.z * t);
      float hs = rs - Rg;
      vec3 sc = betaR * dR(hs) + vec3(betaMs * dM(hs));
      vec3 ex = extinct(hs);
      float muSs = clamp((r * muS + t * dot(d, sun)) / rs, -1.0, 1.0);
      vec3 Ts = texture(uT, ttUv(rs, muSs)).rgb;
      vec3 stepT = exp(-ex * dt);
      // 2nd-order energy (isotropic phase) and the transfer factor
      L2  += T * (1.0 - stepT) * sc / max(ex, vec3(1e-9)) * Ts * 0.0795775;
      fms += T * (1.0 - stepT) * sc / max(ex, vec3(1e-9));
      T *= stepT;
    }
  }
  L2 /= float(ND); fms /= float(ND);
  vec3 psi = L2 / max(vec3(1e-5), 1.0 - fms);
  frag = vec4(psi, 1.0);
}`;

const SKYVIEW_FS = `#version 300 es
precision highp float;
in vec2 vUV; out vec4 frag;
uniform sampler2D uT; uniform sampler2D uMS;
uniform float uCamR; uniform vec3 uUp; uniform vec3 uAzX; uniform vec3 uAzY;
uniform vec3 uSun;
${ATMO_GLSL}
void main() {
  float r0 = uCamR;
  float chor = -ssqrt(r0 * r0 - Rg * Rg) / r0;
  float hor = acos(clamp(chor, -1.0, 1.0));
  float v = vUV.y;
  float theta = v < 0.5
    ? hor * (1.0 - pow(1.0 - 2.0 * v, 2.0))
    : hor + (3.14159265 - hor) * pow(2.0 * v - 1.0, 2.0);
  float phi = (vUV.x - 0.5) * 6.2831853;
  vec3 dir = cos(theta) * uUp + sin(theta) * (cos(phi) * uAzX + sin(phi) * uAzY);
  vec3 pos = uUp * r0;
  // outside the atmosphere: advance to entry, or output vacuum
  float r = r0, mu = dot(dir, uUp);
  if (r > Rt - 1.0) {
    float dTop = -r * mu - ssqrt(r * r * (mu * mu - 1.0) + Rt * Rt);
    if (mu >= 0.0 || r * r * (mu * mu - 1.0) + Rt * Rt < 0.0 || dTop < 0.0) {
      frag = vec4(0.0, 0.0, 0.0, 1.0); return;
    }
    pos += dir * (dTop + 10.0);
  }
  float rS = length(pos);
  float muV = dot(normalize(pos), dir);
  float dg = distToGround(rS, muV);
  float dEnd = dg > 0.0 ? dg : distToTop(rS, muV);
  const int NS = 32;
  float dt = dEnd / float(NS);
  vec3 L = vec3(0.0), T = vec3(1.0);
  for (int i = 0; i < NS; i++) {
    float t = (float(i) + 0.5) * dt;
    vec3 ps = pos + dir * t;
    float rs = length(ps);
    float hs = rs - Rg;
    vec3 upS = ps / rs;
    float muSs = clamp(dot(upS, uSun), -1.0, 1.0);
    float c = dot(dir, uSun);
    vec3 scR = betaR * dR(hs);
    vec3 scM = vec3(betaMs * dM(hs));
    vec3 ex = extinct(hs);
    // sun below the local horizon self-shadows: the transmittance LUT
    // integrates through the planet there and returns ~0
    vec3 Tsun = texture(uT, ttUv(rs, muSs)).rgb;
    vec3 psi = texture(uMS, vec2(muSs * 0.5 + 0.5, (rs - Rg) / (Rt - Rg))).rgb;
    vec3 S = (scR * phR(c) + scM * phM(c)) * Tsun + (scR + scM) * psi;
    vec3 stepT = exp(-ex * dt);
    L += T * (S - S * stepT) / max(ex, vec3(1e-9));
    T *= stepT;
  }
  frag = vec4(L, T.g);
}`;

const HSKY_FS = `#version 300 es
precision highp float;
in vec2 vNdc; out vec4 frag;
uniform vec3 uFwd; uniform vec3 uRight; uniform vec3 uUp;
uniform float uTanHalf; uniform float uAspect;
uniform vec3 uUpW; uniform vec3 uAzX; uniform vec3 uAzY;
uniform vec3 uSun; uniform float uCamR; uniform float uExp;
uniform vec3 uMoon; uniform float uMoonAng; uniform vec3 uMoonZ;
uniform sampler2D uSkyV; uniform sampler2D uT; uniform sampler2D uMoonTex;
${ATMO_GLSL}
void main() {
  vec3 dir = normalize(uFwd + vNdc.x * uTanHalf * uAspect * uRight
                            + vNdc.y * uTanHalf * uUp);
  float theta = acos(clamp(dot(dir, uUpW), -1.0, 1.0));
  float phi = atan(dot(dir, uAzY), dot(dir, uAzX));
  vec4 sv = texture(uSkyV, svUv(uCamR, theta, phi));
  vec3 L = sv.rgb;
  float muV = dot(dir, uUpW);
  float hitG = distToGround(uCamR, muV) > 0.0 ? 0.0 : 1.0;
  // transmittance along the view. A camera above the atmosphere must NOT
  // inherit the in-air clamp: if the ray misses the shell entirely the sun
  // and moon are pure white; if it grazes the limb, evaluate from the true
  // entry point (r*mu + t is invariant along a ray, giving the entry cosine)
  vec3 Tv;
  if (uCamR > Rt - 1.0) {
    float disc = uCamR * uCamR * (muV * muV - 1.0) + Rt * Rt;
    if (muV >= 0.0 || disc < 0.0) Tv = vec3(1.0);
    else {
      float tE = -uCamR * muV - ssqrt(disc);
      float muE = clamp((uCamR * muV + tE) / Rt, -1.0, 1.0);
      Tv = texture(uT, ttUv(Rt - 1.0, muE)).rgb;
    }
  } else Tv = texture(uT, ttUv(uCamR, muV)).rgb;
  // sun: limb-darkened disc + circumsolar halo, added in HDR so the noon
  // disc saturates white through the tonemap and only reddens when the
  // transmittance does
  float ang = acos(clamp(dot(dir, uSun), -1.0, 1.0));
  const float SUNR = 0.004661;                   // 0.267 deg angular radius
  if (ang < SUNR && hitG > 0.0) {
    float x = ang / SUNR;
    float ld = 0.35 + 0.65 * sqrt(max(0.0, 1.0 - x * x));
    L += Tv * ld * 60.0;
  }
  L += hitG * Tv * 0.20 * exp(-pow(ang / (SUNR * 4.0), 1.7));
  // moon: a grey sphere lit by the actual sun direction, so the phase and
  // the crescent's orientation are the real ones; earthshine floor keeps
  // the dark limb faintly visible at night
  float angM = acos(clamp(dot(dir, uMoon), -1.0, 1.0));
  if (angM < uMoonAng && hitG > 0.0) {
    vec3 e1 = normalize(cross(uMoon, abs(uMoon.z) < 0.9 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0)));
    vec3 e2 = cross(uMoon, e1);
    float mu_ = dot(dir, e1) / uMoonAng, mv_ = dot(dir, e2) / uMoonAng;
    float w2 = max(0.0, 1.0 - mu_ * mu_ - mv_ * mv_);
    vec3 n = normalize(e1 * mu_ + e2 * mv_ - uMoon * sqrt(w2));
    float lit = max(dot(n, uSun), 0.0) + 0.006;
    // selenographic mapping: the sub-Earth point is the equirect center
    // (synchronous rotation; the +-8 deg libration wobble is ignored),
    // lunar north approximated by the ecliptic pole (1.5 deg off)
    vec3 mx = -uMoon;
    vec3 my = cross(uMoonZ, mx);
    float slon = atan(dot(n, my), dot(n, mx));
    float slat = asin(clamp(dot(n, uMoonZ), -1.0, 1.0));
    vec3 alb = texture(uMoonTex, vec2(slon / 6.2831853 + 0.5, 0.5 - slat / 3.14159265)).rgb;
    L += Tv * lit * alb * 0.85;
  }
  frag = vec4(vec3(1.0) - exp(-L * uExp), 1.0);
}`;

/* ---------- renderer ---------- */
const GLOBE_BUILD = "2026-08-18b";
class Globe {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { antialias: true, alpha: false,
      depth: true, powerPreference: "high-performance",
      preserveDrawingBuffer: true });   // valid readback for tests/screenshots
    if (!gl) throw new Error("WebGL2 unavailable");
    this.gl = gl;
    this.cam = { lat: 0, lon: 0, alt: 1000, yaw: 0, pitch: -30 };
    this.buildTag = GLOBE_BUILD;
    console.log("globe.js build", GLOBE_BUILD);
    this.tileUrl = opts.tileUrl ||
      (c => `https://tile.openstreetmap.org/${c.z}/${c.x}/${c.y}.png`);
    // Straight from the AWS open-data bucket, which does send
    // Access-Control-Allow-Origin on GET (an earlier HEAD probe suggested
    // otherwise and cost a detour through a local proxy), so the app runs
    // on any static host with no server at all.
    this.hgtUrl = opts.hgtUrl ||
      (c => `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${c.z}/${c.x}/${c.y}.png`);

    this.textures = new Map();   // key -> {tex, frame} | {failed, retryAt}
    this.imgInflight = new Set();
    this.meshes = new Map();     // key -> {vao, vbuf, ibuf, n, anchor, hgtRev}
    this.heights = new Map();    // key -> {grid: Float32Array|null, frame, waiters}
    this.hgtInflight = new Set();
    this.wantHgtUrgent = new Set();
    this.hgtRev = 0;             // bumped on every terrain-tile arrival
    this.lines = new Map();      // name -> slot
    this.markers = new Set();
    this.frame = 0;
    this.refinePx = REFINE_PX;   // adaptive: coarsens when tile counts blow up
    this.lastTiles = 0;
    this.stats = { tiles: 0, texLoads: 0, hgtLoads: 0, rebuilds: 0, buildings: 0 };
    this.wantImg = new Map();    // key -> priority, rebuilt each frame
    this.wantHgt = new Set();
    this.onFatal = opts.onFatal || (e => console.error(e));
    this.bldg = null;            // building-tile state, created on first use

    this._initPrograms();
    this._initStatic();
    this._resize();
    new ResizeObserver(() => this._resize()).observe(canvas);
  }

  _initPrograms() {
    const gl = this.gl;
    const mk = (vs, fs) => {
      const sh = (t, src) => {
        const s = gl.createShader(t);
        gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
          throw new Error(gl.getShaderInfoLog(s) + "\n" + src);
        return s;
      };
      const p = gl.createProgram();
      gl.attachShader(p, sh(gl.VERTEX_SHADER, vs));
      gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
      // pin attribute locations to the slots the VAOs are built against
      // (binding a name a program lacks is harmless and ignored)
      for (const [name, loc] of [["aPos", 0], ["aUV", 1], ["aNrm", 1],
                                 ["aNdc", 0], ["aA", 0], ["aB", 1], ["aSE", 2]])
        gl.bindAttribLocation(p, loc, name);
      gl.linkProgram(p);
      if (!gl.getProgramParameter(p, gl.LINK_STATUS))
        throw new Error(gl.getProgramInfoLog(p));
      const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
      for (let i = 0; i < n; i++) {
        const info = gl.getActiveUniform(p, i);
        u[info.name] = gl.getUniformLocation(p, info.name);
      }
      return { p, u };
    };
    this.pTile = mk(TILE_VS, TILE_FS);
    this.pSky = mk(SKY_VS, SKY_FS);
    this.pStar = mk(STAR_VS, STAR_FS);
    this.pLine = mk(LINE_VS, LINE_FS);
    this.pBldg = mk(BLDG_VS, BLDG_FS);
    this._mk = mk;
  }

  // low-precision solar ephemeris (~0.01 deg) in ECEF, via the same
  // sidereal rotation the stars use
  sunDirEcef(ms) {
    const d = ms / 86400000 - 10957.5;
    const g = (357.529 + 0.98560028 * d) * D2R;
    const lam = (280.459 + 0.98564736 * d
                 + 1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * D2R;
    const e = (23.439 - 0.00000036 * d) * D2R;
    const eq = [Math.cos(lam), Math.cos(e) * Math.sin(lam), Math.sin(e) * Math.sin(lam)];
    const th = this.gmstRad(ms);
    return [Math.cos(th) * eq[0] + Math.sin(th) * eq[1],
            -Math.sin(th) * eq[0] + Math.cos(th) * eq[1], eq[2]];
  }

  // low-precision lunar ephemeris (~0.3 deg): geocentric ECEF position in
  // meters plus angular radius; the caller makes it topocentric (at 60
  // Earth radii, parallax is up to a degree and matters)
  moonEcef(ms) {
    const d = ms / 86400000 - 10957.5;
    const L = (218.316 + 13.176396 * d) * D2R;
    const M = (134.963 + 13.064993 * d) * D2R;
    const F = (93.272 + 13.229350 * d) * D2R;
    const Dm = (297.850 + 12.190749 * d) * D2R;
    const Ms = (357.529 + 0.985600 * d) * D2R;
    const lam = L + D2R * (6.289 * Math.sin(M) - 1.274 * Math.sin(M - 2 * Dm)
      + 0.658 * Math.sin(2 * Dm) - 0.214 * Math.sin(2 * M) - 0.186 * Math.sin(Ms)
      - 0.114 * Math.sin(2 * F));
    const bet = D2R * 5.128 * Math.sin(F);
    const dist = 385001 - 20905 * Math.cos(M) - 3699 * Math.cos(2 * Dm - M)
      - 2956 * Math.cos(2 * Dm);                       // km
    const e = (23.439 - 0.00000036 * d) * D2R;
    const x = Math.cos(bet) * Math.cos(lam);
    const y = Math.cos(e) * Math.cos(bet) * Math.sin(lam) - Math.sin(e) * Math.sin(bet);
    const z = Math.sin(e) * Math.cos(bet) * Math.sin(lam) + Math.cos(e) * Math.sin(bet);
    const th = this.gmstRad(ms);
    const ex = Math.cos(th) * x + Math.sin(th) * y;
    const ey = -Math.sin(th) * x + Math.cos(th) * y;
    const m = dist * 1000;
    return { pos: [ex * m, ey * m, z * m], ang: Math.asin(1737.4 / dist) };
  }

  // the five naked-eye planets: J2000 Keplerian elements + century rates
  // (JPL approximate table), geocentric equatorial unit vectors plus
  // apparent magnitude (classic Astronomical Almanac phase formulas) and a
  // B-V-ish tint for the star pipeline, which rotates them by GMST like
  // any other star
  planetsEq(ms) {
    const d = ms / 86400000 - 10957.5, T = d / 36525;
    const EL = [
      [0.38709927, 0.20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593,
       0.00000037, 0.00001906, -0.00594749, 149472.67411175, 0.16047689, -0.12534081],
      [0.72333566, 0.00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255,
       0.00000390, -0.00004107, -0.00078890, 58517.81538729, 0.00268329, -0.27769418],
      [1.00000261, 0.01671123, -0.00001531, 100.46457166, 102.93768193, 0.0,
       0.00000562, -0.00004392, -0.01294668, 35999.37244981, 0.32327364, 0.0],
      [1.52371034, 0.09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891,
       0.00001847, 0.00007882, -0.00813131, 19140.30268499, 0.44441088, -0.29257343],
      [5.20288700, 0.04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909,
       -0.00011607, -0.00013253, -0.00183714, 3034.74612775, 0.21252668, 0.20469106],
      [9.53667594, 0.05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448,
       -0.00125060, -0.00050991, 0.00193609, 1222.49362201, -0.41897216, -0.28867794],
    ];
    const helio = i => {
      const E0 = EL[i];
      const a = E0[0] + E0[6] * T, ec = E0[1] + E0[7] * T, I = (E0[2] + E0[8] * T) * D2R;
      const Lg = E0[3] + E0[9] * T, wb = E0[4] + E0[10] * T, Om = E0[5] + E0[11] * T;
      let M = ((Lg - wb) % 360) * D2R;
      let E = M;
      for (let k = 0; k < 6; k++) E -= (E - ec * Math.sin(E) - M) / (1 - ec * Math.cos(E));
      const xp = a * (Math.cos(E) - ec), yp = a * Math.sqrt(1 - ec * ec) * Math.sin(E);
      const w = (wb - Om) * D2R, o = Om * D2R;
      const cw = Math.cos(w), sw = Math.sin(w), co = Math.cos(o), so = Math.sin(o);
      const ci = Math.cos(I), si = Math.sin(I);
      return [
        (cw * co - sw * so * ci) * xp + (-sw * co - cw * so * ci) * yp,
        (cw * so + sw * co * ci) * xp + (-sw * so + cw * co * ci) * yp,
        sw * si * xp + cw * si * yp,
      ];
    };
    const eaC = helio(2);
    const Re = Math.hypot(eaC[0], eaC[1], eaC[2]);
    const eps = (23.439 - 0.00000036 * d) * D2R;
    const MAG = [
      [-0.42, ph => 0.0380 * ph - 0.000273 * ph * ph + 0.000002 * ph ** 3, 0.93],
      [-4.40, ph => 0.0009 * ph + 0.000239 * ph * ph - 0.00000065 * ph ** 3, 0.82],
      null,
      [-1.52, ph => 0.016 * ph, 1.36],
      [-9.40, ph => 0.005 * ph, 0.83],
      [-8.88, ph => 0.044 * ph, 1.04],
    ];
    const out = [];
    for (const i of [0, 1, 3, 4, 5]) {
      const p = helio(i);
      const g = [p[0] - eaC[0], p[1] - eaC[1], p[2] - eaC[2]];
      const r = Math.hypot(p[0], p[1], p[2]);
      const del = Math.hypot(g[0], g[1], g[2]);
      const cph = Math.min(1, Math.max(-1, (r * r + del * del - Re * Re) / (2 * r * del)));
      const ph = Math.acos(cph) / D2R;
      const mag = MAG[i][0] + 5 * Math.log10(r * del) + MAG[i][1](ph);
      const eq = [g[0] / del,
        (Math.cos(eps) * g[1] - Math.sin(eps) * g[2]) / del,
        (Math.sin(eps) * g[1] + Math.cos(eps) * g[2]) / del];
      out.push({ eq, mag, bv: MAG[i][2] });
    }
    return out;
  }

  _initAtmo() {
    const gl = this.gl;
    this.atmoOK = !!gl.getExtension("EXT_color_buffer_float");
    this.skyMode = this.atmoOK ? "hillaire" : "analytic";
    if (!this.atmoOK) return;
    const mk = this._mk;
    this.pTrans = mk(AQUAD_VS, TRANS_FS);
    this.pMulti = mk(AQUAD_VS, MULTI_FS);
    this.pSkyView = mk(AQUAD_VS, SKYVIEW_FS);
    this.pSkyH = mk(SKY_VS, HSKY_FS);
    const t2 = (w, h) => {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    };
    this.tTrans = t2(256, 64);
    this.tMulti = t2(32, 32);
    this.tSkyV = t2(192, 108);
    this.aFbo = gl.createFramebuffer();
    // the two static LUTs, once
    this._lutPass(this.pTrans, this.tTrans, 256, 64, () => {});
    this._lutPass(this.pMulti, this.tMulti, 32, 32, u => {
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.tTrans);
      gl.uniform1i(u.uT, 6);
    });
  }

  _lutPass(prog, tex, w, h, setU, layer) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.aFbo);
    if (layer === undefined)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    else
      gl.framebufferTextureLayer(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, tex, 0, layer);
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false); gl.disable(gl.BLEND);
    gl.useProgram(prog.p);
    setU(prog.u);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.depthMask(true);
  }

  _buildSkyLUTs() {
    const gl = this.gl;
    const P = this.camPos;
    const camR = Math.hypot(P[0], P[1], P[2]);
    const up = [P[0] / camR, P[1] / camR, P[2] / camR];
    const sun = this.sunDirEcef(Date.now());
    this.sunNow = sun; this.camRNow = camR;
    const sdu = sun[0] * up[0] + sun[1] * up[1] + sun[2] * up[2];
    this.sunElevNow = sdu;
    const mn = this.moonEcef(Date.now());
    const mvx = mn.pos[0] - P[0], mvy = mn.pos[1] - P[1], mvz = mn.pos[2] - P[2];
    const md = Math.hypot(mvx, mvy, mvz);
    const mdir = [mvx / md, mvy / md, mvz / md];
    // lunar north ~ ecliptic pole, orthogonalized against the sub-Earth axis
    const th2 = this.gmstRad(Date.now());
    const eps2 = 23.439 * D2R;
    const pole = [-Math.sin(th2) * Math.sin(eps2), -Math.cos(th2) * Math.sin(eps2), Math.cos(eps2)];
    const mX = [-mdir[0], -mdir[1], -mdir[2]];
    const pd = pole[0] * mX[0] + pole[1] * mX[1] + pole[2] * mX[2];
    let mZ = [pole[0] - pd * mX[0], pole[1] - pd * mX[1], pole[2] - pd * mX[2]];
    const zn = Math.hypot(mZ[0], mZ[1], mZ[2]);
    mZ = [mZ[0] / zn, mZ[1] / zn, mZ[2] / zn];
    this.moonNow = { dir: mdir, ang: mn.ang, z: mZ };
    let ax = [sun[0] - sdu * up[0], sun[1] - sdu * up[1], sun[2] - sdu * up[2]];
    let n = Math.hypot(ax[0], ax[1], ax[2]);
    if (n < 1e-6) { ax = Math.abs(up[2]) < 0.9 ? [ -up[1], up[0], 0 ] : [0, -up[2], up[1]]; n = Math.hypot(ax[0], ax[1], ax[2]); }
    ax = [ax[0] / n, ax[1] / n, ax[2] / n];
    const ay = [up[1] * ax[2] - up[2] * ax[1], up[2] * ax[0] - up[0] * ax[2], up[0] * ax[1] - up[1] * ax[0]];
    this.svFrame = { up, ax, ay };
    const common = u => {
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.tTrans);
      gl.uniform1i(u.uT, 6);
      gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.tMulti);
      gl.uniform1i(u.uMS, 7);
      gl.uniform1f(u.uCamR, camR);
      gl.uniform3fv(u.uUp, up);
      gl.uniform3fv(u.uAzX, ax);
      gl.uniform3fv(u.uAzY, ay);
      gl.uniform3fv(u.uSun, sun);
    };
    this._lutPass(this.pSkyView, this.tSkyV, 192, 108, common);
    // the froxel volume is gone: terrain aerial now derives from the
    // sky-view LUT + closed-form opacity in the tile shader (the 3D-texture
    // path misrendered on real GPUs while passing on SwiftShader)
  }

  async _loadStars() {
    try {
      const buf = await (await fetch("stars.bin")).arrayBuffer();
      const n = new Uint32Array(buf, 0, 1)[0];
      const f = new Float32Array(buf, 4);
      const out = new Float32Array(n * 5);
      for (let i = 0; i < n; i++) {
        const ra = f[i * 4] * D2R, dec = f[i * 4 + 1] * D2R;
        out[i * 5]     = Math.cos(dec) * Math.cos(ra);
        out[i * 5 + 1] = Math.cos(dec) * Math.sin(ra);
        out[i * 5 + 2] = Math.sin(dec);
        out[i * 5 + 3] = f[i * 4 + 2];
        out[i * 5 + 4] = f[i * 4 + 3];
      }
      const gl = this.gl;
      this.starBuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.starBuf);
      gl.bufferData(gl.ARRAY_BUFFER, out, gl.STATIC_DRAW);
      const p = this.pStar.p;
      this.starLoc = { dir: gl.getAttribLocation(p, "aDir"),
                       mag: gl.getAttribLocation(p, "aMag"),
                       bv:  gl.getAttribLocation(p, "aBv") };
      this.starCount = n;
    } catch (e) { this.starCount = 0; }   // no stars is a graceful state
  }
  // Greenwich mean sidereal time, radians, from a unix-ms clock
  gmstRad(ms) {
    const d = ms / 86400000 - 10957.5;          // days since J2000.0
    const deg = (280.46061837 + 360.98564736629 * d) % 360;
    return (deg < 0 ? deg + 360 : deg) * D2R;
  }

  _initStatic() {
    const gl = this.gl;
    // fullscreen triangle for the sky
    this.skyBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    this.starCount = 0;
    this._loadStars();
    this._initAtmo();
    {  // the real face: NASA LROC albedo map, 1024x512 (moon.jpg)
      const img = new Image();
      img.onload = () => {
        const t = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, t);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        this.moonTex = t;
      };
      img.src = "moon.jpg";
    }
    // 1x1 fallback texture (paper tone) so a tile can always draw; the
    // magenta twin exists for tests only (OSM's own land colors sit within
    // classifier range of the paper tone, so "paper %" over-counted)
    this.paperTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.paperTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([233, 229, 221, 255]));
    this.debugTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.debugTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([255, 0, 255, 255]));
    this.debugPaper = false;
    // polar caps: fans over the latitudes mercator tiles cannot reach,
    // tucked 1 km under the tile edge so no seam shows
    this.caps = [1, -1].map(sgn => {
      const ring = [], N = 48, ringLat = sgn * 84.9;
      const anchor = ecef(sgn * 90, 0, 0);
      const vs = [0, 0, 0];
      for (let i = 0; i <= N; i++) {
        const p = vsub(ecef(ringLat, i / N * 360 - 180, -1000), anchor);
        vs.push(p[0], p[1], p[2]);
      }
      const is = [];
      for (let i = 1; i <= N; i++) is.push(0, i, i + 1);
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      const vb = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, vb);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vs), gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
      const ib = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(is), gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      const color = sgn > 0 ? [0.83, 0.88, 0.92, 1] : [0.93, 0.95, 0.97, 1];
      return { vao, n: is.length, anchor, color };
    });
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
    this.dpr = dpr;
  }

  /* ---------- camera ---------- */
  _updateCamera() {
    const c = this.cam;
    this.camPos = ecef(c.lat, c.lon, c.alt);           // doubles
    const la = c.lat * D2R, lo = c.lon * D2R;
    const up = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
    const east = vunit([-up[1], up[0], 0]);
    const north = vcross(up, east);
    const yr = c.yaw * D2R, pr = c.pitch * D2R, cp = Math.cos(pr);
    const fwd = [
      east[0] * Math.sin(yr) * cp + north[0] * Math.cos(yr) * cp + up[0] * Math.sin(pr),
      east[1] * Math.sin(yr) * cp + north[1] * Math.cos(yr) * cp + up[1] * Math.sin(pr),
      east[2] * Math.sin(yr) * cp + north[2] * Math.cos(yr) * cp + up[2] * Math.sin(pr)];
    const right = vunit(vcross(fwd, up));
    const cUp = vcross(right, fwd);
    this.camFwd = fwd; this.camRight = right; this.camUp = cUp; this.camUpWorld = up;
    // view rotation (rows right / up / -fwd), then projection
    const aspect = this.canvas.width / this.canvas.height;
    this.aspect = aspect;
    this.tanHalf = Math.tan(FOVY / 2);
    const f = 1 / this.tanHalf, near = 0.5, far = LOG_FAR;
    // column-major P * R, rotation only (translation is done CPU-side)
    const R = right, U = cUp, F = fwd;
    const P = [f / aspect, 0, 0, 0, 0, f, 0, 0,
               0, 0, (far + near) / (near - far), -1,
               0, 0, 2 * far * near / (near - far), 0];
    const V = [R[0], U[0], -F[0], 0, R[1], U[1], -F[1], 0, R[2], U[2], -F[2], 0, 0, 0, 0, 1];
    this.vp = mat4mul(P, V);
    this.logF = 2 / Math.log2(LOG_FAR + 1);
    const agl = Math.max(c.alt - (this.terrainAt(c.lat, c.lon) ?? 0), 2);
    this.horizon = Math.sqrt(Math.max(2 * GLOBE_R * agl + agl * agl, 1));
    this.camAglNow = agl;
    // Haze distances: at altitude the horizon sets them, but a ground-level
    // observer's 5 km geometric horizon must not become a 5 km fog wall
    // (mountains 60 km off are genuinely visible from a field); floor them.
    this.fogFar = Math.max(this.horizon * 1.05, 65000);
    this.fogNear = this.fogFar * 0.5;
  }

  /* ---------- terrain (terrarium) ---------- */
  _hgtKeyFor(lat, lon, z = TERRAIN_Z) {
    const x = Math.min((1 << z) - 1, Math.max(0, Math.floor(lon2tx(lon, z))));
    const y = Math.min((1 << z) - 1, Math.max(0, Math.floor(lat2ty(lat, z))));
    return { z, x, y, key: tkey(z, x, y) };
  }
  _requestHgt(z, x, y, urgent) {
    const key = tkey(z, x, y);
    let e = this.heights.get(key);
    if (e) {
      e.frame = this.frame;
      // an explicit caller can promote a queued scenery request
      if (urgent && !e.grid && !e.loading && this.wantHgt.has(key))
        this.wantHgtUrgent.add(key);
      // a failed fetch becomes retryable after a short backoff; without
      // this, one transient network error blackholed the tile for the whole
      // session and the flight hold ("Waiting for terrain data...") never
      // released
      if (e.failed && !e.loading && Date.now() - e.failedAt > HGT_RETRY_MS) {
        e.failed = false;
        e.promise = new Promise(res => e.waiters.push(res));
        (urgent ? this.wantHgtUrgent : this.wantHgt).add(key);
      }
      return e;
    }
    e = { grid: null, frame: this.frame, waiters: [], promise: null };
    e.promise = new Promise(res => e.waiters.push(res));
    this.heights.set(key, e);
    (urgent ? this.wantHgtUrgent : this.wantHgt).add(key);
    return e;
  }
  _pumpHgt() {
    // urgent first: ensureTerrain callers (a launch pad, a landing check)
    // must not queue behind dozens of scenery-mesh requests; that FIFO wait
    // was most of the falcon9 click-to-liftoff delay
    for (const key of [...this.wantHgtUrgent, ...this.wantHgt]) {
      if (this.hgtInflight.size >= MAX_HGT_INFLIGHT) break;
      this.wantHgtUrgent.delete(key);
      this.wantHgt.delete(key);
      const e = this.heights.get(key);
      if (!e || e.grid || e.loading) continue;
      e.loading = true;
      this.hgtInflight.add(key);
      const [z, x, y] = key.split("/").map(Number);
      // Raw float32 heights from the server, decoded there from the PNG.
      // Deliberately NOT an Image + canvas + getImageData path: Brave's
      // fingerprint shield adds seeded +-1 noise to canvas readbacks, and
      // +-1 in terrarium's red channel is +-256 m: deterministic pyramids
      // in the terrain that only existed in that browser. fetch() of raw
      // bytes is not a fingerprinting surface, so it arrives exact.
      const done = () => {
        e.loading = false;
        this.hgtInflight.delete(key);
        for (const w of e.waiters) w();
        e.waiters = [];
      };
      fetch(this.hgtUrl({ z, x, y }))
        .then(r => r.ok ? r.arrayBuffer() : Promise.reject(r.status))
        .then(decodeTerrarium)
        .then(g => {
          if (g.length !== 65536) throw new Error("bad tile size");
          // sanitize: the bucket contains occasional corrupt patches
          // (measured: 696 pixels swinging the full +-32768 m inside an
          // otherwise flat Lake Erie tile); out-of-band values are nodata,
          // filled with the tile's valid mean
          let sum = 0, cnt = 0;
          for (let i = 0; i < 65536; i++) {
            const h = g[i];
            if (h >= -1000 && h <= 9200) { sum += h; cnt++; } else g[i] = NaN;
          }
          if (cnt < 65536) {
            const fill = cnt ? sum / cnt : 0;
            for (let i = 0; i < 65536; i++) if (Number.isNaN(g[i])) g[i] = fill;
          }
          e.grid = g;
          this.hgtRev++;
          this.stats.hgtLoads++;
          done();
        })
        .catch(() => { e.failed = true; e.failedAt = Date.now(); done(); });
    }
    // LRU
    if (this.heights.size > HGT_CAP) {
      const ent = [...this.heights.entries()].filter(([, e]) => e.grid)
        .sort((a, b) => a[1].frame - b[1].frame);
      for (let i = 0; i < ent.length - HGT_CAP + 40; i++) this.heights.delete(ent[i][0]);
    }
  }
  // bilinear sample of one specific terrain tile's grid, edge-clamped: a
  // point exactly on (or just past) the tile boundary reads the edge pixel
  // instead of resolving to a neighboring tile
  _sampleGrid(g, z, x, y, lat, lon) {
    const fx = (lon2tx(lon, z) - x) * 256 - 0.5, fy = (lat2ty(lat, z) - y) * 256 - 0.5;
    const x0 = Math.max(0, Math.min(255, Math.floor(fx)));
    const y0 = Math.max(0, Math.min(255, Math.floor(fy)));
    const x1 = Math.min(255, x0 + 1), y1 = Math.min(255, y0 + 1);
    const tx = Math.max(0, Math.min(1, fx - x0)), ty = Math.max(0, Math.min(1, fy - y0));
    return (g[y0 * 256 + x0] * (1 - tx) + g[y0 * 256 + x1] * tx) * (1 - ty)
         + (g[y1 * 256 + x0] * (1 - tx) + g[y1 * 256 + x1] * tx) * ty;
  }
  // height above MSL from the deepest loaded terrarium tile; null before data
  terrainAt(lat, lon) {
    if (lat > MAXLAT || lat < -MAXLAT) return 0;
    for (let z = TERRAIN_Z; z >= 0; z--) {
      const x = Math.floor(lon2tx(lon, z)), y = Math.floor(lat2ty(lat, z));
      const e = this.heights.get(tkey(z, x, y));
      if (!e || !e.grid) continue;
      e.frame = this.frame;
      return this._sampleGrid(e.grid, z, x, y, lat, lon);
    }
    return null;
  }
  // state of the z12 tile under a point, for honest progress hints:
  // "loaded" | "loading" | "failed" (in backoff) | "none" (never requested)
  terrainState(lat, lon) {
    const { z, x, y } = this._hgtKeyFor(lat, lon);
    const e = this.heights.get(tkey(z, x, y));
    if (!e) return "none";
    if (e.grid) return "loaded";
    return e.failed ? "failed" : "loading";
  }
  // the ONE terrain tile a z/x/y imagery tile's mesh samples (quadtree-
  // aligned, so the imagery tile lies entirely inside it), plus the exact
  // rebuild stamp: tile identity + loaded state. A mesh goes stale exactly
  // when this string changes, and in no other case; the old scheme stamped
  // only the deepest loaded tile at the CENTER, so a mesh whose edge
  // vertices had fallen back to a coarse grid (wrong by up to ~100 m on
  // steep relief: the terrain-pyramid bug) never rebuilt.
  _hgtTileFor(z, x, y) {
    const tz = Math.min(z, TERRAIN_Z);
    const tx = x >> (z - tz), ty = y >> (z - tz);
    return { tz, tx, ty, key: tkey(tz, tx, ty) };
  }
  _hgtStampFor(z, x, y) {
    if (z < 8) return "flat";
    const T = this._hgtTileFor(z, x, y);
    const e = this.heights.get(T.key);
    return T.key + ":" + (e && e.grid ? 1 : 0);
  }
  // make sure terrarium tiles cover these [lat, lon] points; resolves loaded
  ensureTerrain(points) {
    const ps = [];
    for (const [la, lo] of points) {
      if (Math.abs(la) > 85) continue;
      const { z, x, y } = this._hgtKeyFor(la, lo);
      ps.push(this._requestHgt(z, x, y, true).promise);
    }
    this._pumpHgt();
    return Promise.all(ps);
  }

  /* ---------- tile meshes ---------- */
  _tileGeo(z, x, y) {
    const latN = ty2lat(y, z), latS = ty2lat(y + 1, z);
    const lonW = tx2lon(x, z), lonE = tx2lon(x + 1, z);
    const latM = (latN + latS) / 2, lonM = (lonW + lonE) / 2;
    const span = 2 * Math.PI * GLOBE_R * Math.cos(latM * D2R) / (1 << z);
    return { latN, latS, lonW, lonE, latM, lonM, span };
  }
  _buildMesh(z, x, y) {
    const gl = this.gl;
    const geo = this._tileGeo(z, x, y);
    const anchor = ecef(geo.latM, geo.lonM, 0);
    const G = z < 4 ? 16 : 8;
    const useHgt = z >= 8;
    let grid = null;
    let T = null;
    if (useHgt) {              // every vertex samples this ONE terrain tile
      T = this._hgtTileFor(z, x, y);
      const e = this._requestHgt(T.tz, T.tx, T.ty);
      grid = e.grid || null;
    }
    const vs = [], is = [];
    const V = (i, j, drop) => {         // grid vertex (i: x/east, j: y/south)
      const ty_ = y + j / G, tx_ = x + i / G;
      const lat = ty2lat(ty_, z), lon = tx2lon(tx_, z);
      // pinned-tile sampling; until the tile arrives, the multi-zoom walker
      // is a smooth placeholder, and the stamp flip rebuilds this mesh the
      // moment the real tile loads
      let h = !useHgt ? 0
        : grid ? this._sampleGrid(grid, T.tz, T.tx, T.ty, lat, lon)
        : (this.terrainAt(lat, lon) ?? 0);
      h -= drop;
      const p = vsub(ecef(lat, lon, h), anchor);
      vs.push(p[0], p[1], p[2], i / G, j / G);
    };
    for (let j = 0; j <= G; j++) for (let i = 0; i <= G; i++) V(i, j, 0);
    for (let j = 0; j < G; j++)
      for (let i = 0; i < G; i++) {
        const a = j * (G + 1) + i, b = a + 1, c = a + G + 1, d = c + 1;
        is.push(a, c, b, b, c, d);
      }
    // skirts: one dropped copy of each border vertex, stitched to the border
    const drop = geo.span * 0.08 + 40;
    const border = [];
    for (let i = 0; i <= G; i++) border.push([i, 0]);
    for (let j = 1; j <= G; j++) border.push([G, j]);
    for (let i = G - 1; i >= 0; i--) border.push([i, G]);
    for (let j = G - 1; j >= 1; j--) border.push([0, j]);
    const skirtBase = (G + 1) * (G + 1);
    for (const [i, j] of border) V(i, j, drop);
    for (let k = 0; k < border.length; k++) {
      const [i1, j1] = border[k], [i2, j2] = border[(k + 1) % border.length];
      const t1 = j1 * (G + 1) + i1, t2 = j2 * (G + 1) + i2;
      const s1 = skirtBase + k, s2 = skirtBase + (k + 1) % border.length;
      is.push(t1, s1, t2, t2, s1, s2);
    }
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const vb = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vb);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vs), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 20, 0);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 20, 12);
    const ib = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(is), gl.STATIC_DRAW);
    gl.bindVertexArray(null);
    return { vao, vb, ib, n: is.length, anchor, geo,
             hgtStamp: this._hgtStampFor(z, x, y),
             frame: this.frame };
  }
  _mesh(z, x, y) {
    const key = tkey(z, x, y);
    let m = this.meshes.get(key);
    if (m && m.hgtStamp !== "flat" && this.rebuildBudget > 0
        && this._hgtStampFor(z, x, y) !== m.hgtStamp) {
      this.rebuildBudget--;
      this.stats.rebuilds++;
      this._disposeMesh(m);
      m = null;
    }
    if (!m) { m = this._buildMesh(z, x, y); this.meshes.set(key, m); }
    m.frame = this.frame;
    return m;
  }
  _disposeMesh(m) {
    const gl = this.gl;
    gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.vb); gl.deleteBuffer(m.ib);
  }

  /* ---------- imagery textures ---------- */
  _texFor(z, x, y) {          // best loaded texture at-or-above this tile
    let cz = z, cx = x, cy = y;
    while (cz >= 0) {
      const e = this.textures.get(tkey(cz, cx, cy));
      if (e && e.tex) {
        e.frame = this.frame;
        const dz = z - cz, s = 1 / (1 << dz);
        return { tex: e.tex, uvt: [s, s, (x - cx * (1 << dz)) * s, (y - cy * (1 << dz)) * s] };
      }
      cz--; cx >>= 1; cy >>= 1;
    }
    return { tex: this.debugPaper ? this.debugTex : this.paperTex, uvt: [1, 1, 0, 0] };
  }
  _wantTex(z, x, y, priority) {
    const key = tkey(z, x, y);
    const e = this.textures.get(key);
    if (e && (e.tex || e.loading)) {
      if (e.tex) return;                 // loaded: nothing to want
    } else if (!(e && e.failed && performance.now() < e.retryAt)) {
      const cur = this.wantImg.get(key);
      if (cur == null || priority > cur) this.wantImg.set(key, priority);
    }
    // coarse-first: if the parent is missing too, one parent download covers
    // four children while the fine tiles stream in behind it
    if (z > BASE_Z) {
      const pk = tkey(z - 1, x >> 1, y >> 1);
      const pe = this.textures.get(pk);
      if (!pe || (!pe.tex && !pe.loading)) this._wantTex(z - 1, x >> 1, y >> 1, priority * 3);
    }
  }
  _pumpImg() {
    const cands = [...this.wantImg.entries()].sort((a, b) => b[1] - a[1]);
    this.wantImg.clear();
    const gl = this.gl;
    for (const [key] of cands) {
      if (this.imgInflight.size >= MAX_IMG_INFLIGHT) break;
      let e = this.textures.get(key);
      if (e && (e.tex || e.loading)) continue;
      if (!e) { e = {}; this.textures.set(key, e); }
      e.loading = true;
      this.imgInflight.add(key);
      const [z, x, y] = key.split("/").map(Number);
      const img = new Image();
      img.crossOrigin = "anonymous";
      img.onload = () => {
        this.imgInflight.delete(key);
        e.loading = false;
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        e.tex = tex; e.frame = this.frame;
        this.stats.texLoads++;
      };
      img.onerror = () => {
        this.imgInflight.delete(key);
        e.loading = false; e.failed = true;
        e.retryAt = performance.now() + 6000;
      };
      img.src = this.tileUrl({ z, x, y });
    }
    // LRU eviction. The caps must FLOAT ABOVE the live working set: with a
    // fixed 700-entry cap and 955 visible tiles, every frame rebuilt ~335
    // meshes and re-downloaded their textures, the eviction pass deleted
    // them again (all stamped the same frame, so the sort could not even
    // prefer survivors), and downtown performance collapsed into cache
    // thrash (measured: 334 mesh deletes per frame, all from this site).
    // Entries touched THIS frame are never eligible.
    const cap = Math.max(TEX_CAP, Math.ceil(this.lastTiles * 1.6));
    const floor = Math.max(TEX_CAP - 80, Math.ceil(this.lastTiles * 1.3));
    if (this.textures.size > cap) {
      const ent = [...this.textures.entries()]
        .filter(([k, e]) => e.tex && e.frame !== this.frame && Number(k.split("/")[0]) > BASE_Z)
        .sort((a, b) => a[1].frame - b[1].frame);
      for (let i = 0; i < Math.min(ent.length, this.textures.size - floor); i++) {
        this.gl.deleteTexture(ent[i][1].tex);
        this.textures.delete(ent[i][0]);
      }
    }
    if (this.meshes.size > cap) {
      const ent = [...this.meshes.entries()]
        .filter(([, m]) => m.frame !== this.frame)
        .sort((a, b) => a[1].frame - b[1].frame);
      for (let i = 0; i < Math.min(ent.length, this.meshes.size - floor); i++) {
        this._disposeMesh(ent[i][1]);
        this.meshes.delete(ent[i][0]);
      }
    }
  }

  /* ---------- visibility + traversal ---------- */
  // Bounding sphere: centered at the tile's real terrain height when known
  // (else height 0 with a 9 km pad, so unloaded mountain tiles are never
  // wrongly culled), radius = half-diagonal + a relief allowance. The pad is
  // deliberately NOT in the screen-size estimate: an inflated radius floored
  // the distance term and refined everything near the camera to max depth
  // (measured: 8,051 leaves from 900 m; the texture loader can never win).
  _tileVisible(z, x, y) {
    const geo = this._tileGeo(z, x, y);
    const hC = z >= 8 ? this.terrainAt(geo.latM, geo.lonM) : 0;
    const c = ecef(geo.latM, geo.lonM, hC ?? 0);
    const pad = hC == null ? 9000 : Math.min(geo.span * 0.75 + 150, 6000);
    const rad = geo.span * 0.75 + pad;
    // horizon cull on the sphere
    const camR = Math.max(vnorm(this.camPos), GLOBE_R + 2);
    const horizAng = Math.acos(Math.min(1, GLOBE_R / camR));
    const cosSep = vdot(vunit(c), vunit(this.camPos));
    if (cosSep < Math.cos(Math.min(horizAng + rad / GLOBE_R + 0.05, Math.PI))) return null;
    const rel = vsub(c, this.camPos);
    const dist = vnorm(rel);
    if (dist - rad > this.fogFar * 1.6) return null;   // fully fogged out
    // exact frustum-sphere test: near plane, then the four side planes
    const dF = vdot(rel, this.camFwd);
    if (dF < -rad) return null;
    const tx = this.tanHalf * this.aspect, ty = this.tanHalf;
    const nx = 1 / Math.sqrt(1 + tx * tx), ny = 1 / Math.sqrt(1 + ty * ty);
    const dR = vdot(rel, this.camRight), dU = vdot(rel, this.camUp);
    if ((dR + tx * dF) * nx < -rad) return null;       // left
    if ((-dR + tx * dF) * nx < -rad) return null;      // right
    if ((dU + ty * dF) * ny < -rad) return null;       // bottom
    if ((-dU + ty * dF) * ny < -rad) return null;      // top
    return { geo, dist, geoRad: geo.span * 0.75 };
  }
  _traverse() {
    const out = [];
    const pxPerM = this.canvas.height / (2 * this.tanHalf);   // at 1 m distance
    const agl = this.camAglNow;
    const visit = (z, x, y) => {
      const v = this._tileVisible(z, x, y);
      if (!v) return;
      const den = Math.max(v.dist - v.geoRad, agl * 0.7, 20);
      const px = v.geo.span / den * pxPerM;
      if (px > this.refinePx && z < MAX_Z) {
        visit(z + 1, 2 * x, 2 * y); visit(z + 1, 2 * x + 1, 2 * y);
        visit(z + 1, 2 * x, 2 * y + 1); visit(z + 1, 2 * x + 1, 2 * y + 1);
      } else out.push({ z, x, y, px, dist: v.dist });
    };
    const n = 1 << BASE_Z;
    for (let x = 0; x < n; x++) for (let y = 0; y < n; y++) visit(BASE_Z, x, y);
    return out;
  }

  /* ---------- overlays ---------- */
  // polys: array of polylines, each an array of [lat, lon, hMSL]
  setLines(name, polys, color, width) {
    let s = this.lines.get(name);
    if (!s) { s = { vbuf: null, ibuf: null, vao: null, n: 0 }; this.lines.set(name, s); }
    s.polys = polys; s.color = color; s.width = width; s.dirty = true;
  }
  clearLines(name) {
    const s = this.lines.get(name);
    if (s) { s.polys = []; s.dirty = true; s.n = 0; }
  }
  _buildLines(s) {
    const gl = this.gl;
    s.dirty = false;
    if (!s.polys || !s.polys.length || !s.polys[0].length) { s.n = 0; return; }
    const p0 = s.polys[0][0];
    s.anchor = ecef(p0[0], p0[1], p0[2]);
    const vs = [], is = [];
    let vi = 0;
    for (const poly of s.polys) {
      const loc = poly.map(p => vsub(ecef(p[0], p[1], p[2]), s.anchor));
      for (let i = 0; i + 1 < loc.length; i++) {
        const a = loc[i], b = loc[i + 1];
        for (const [side, end] of [[-1, 0], [1, 0], [-1, 1], [1, 1]])
          vs.push(a[0], a[1], a[2], b[0], b[1], b[2], side, end);
        is.push(vi, vi + 1, vi + 2, vi + 1, vi + 3, vi + 2);
        vi += 4;
      }
    }
    if (!s.vao) {
      s.vao = gl.createVertexArray();
      s.vbuf = gl.createBuffer(); s.ibuf = gl.createBuffer();
      gl.bindVertexArray(s.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, s.vbuf);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 32, 0);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 32, 12);
      gl.enableVertexAttribArray(2);
      gl.vertexAttribPointer(2, 2, gl.FLOAT, false, 32, 24);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, s.ibuf);
      gl.bindVertexArray(null);
    }
    gl.bindVertexArray(s.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, s.vbuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vs), gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, s.ibuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(is), gl.DYNAMIC_DRAW);
    gl.bindVertexArray(null);
    s.n = is.length;
  }

  addMarker(el, opts = {}) {
    const m = { el, lat: 0, lon: 0, h: 0, anchor: opts.anchor || "center",
      set: false, globe: this };
    m.setPos = (lat, lon, h) => { m.lat = lat; m.lon = lon; m.h = h ?? m.h; m.set = true; };
    m.remove = () => { this.markers.delete(m); if (el.parentNode) el.parentNode.removeChild(el); };
    this.markers.add(m);
    return m;
  }
  _placeMarkers() {
    for (const m of this.markers) {
      if (!m.set) { m.el.style.visibility = "hidden"; continue; }
      const pr = this.project(m.lat, m.lon, m.h);
      if (!pr || !pr.visible) { m.el.style.visibility = "hidden"; continue; }
      // "visible", never "": clearing the inline value hands control back
      // to the .gmk class default (hidden), which kept every marker
      // invisible while the math placed them perfectly
      m.el.style.visibility = "visible";
      m.el.style.transform = `translate(${pr.x}px, ${pr.y}px) translate(-50%, ${m.anchor === "bottom" ? "-100%" : "-50%"})`;
    }
  }

  // world -> CSS pixel projection; null/hidden when behind the camera
  project(lat, lon, h) {
    const p = vsub(ecef(lat, lon, h), this.camPos);
    const m = this.vp;
    const cx = m[0] * p[0] + m[4] * p[1] + m[8] * p[2];
    const cy = m[1] * p[0] + m[5] * p[1] + m[9] * p[2];
    const cw = m[3] * p[0] + m[7] * p[1] + m[11] * p[2];
    if (cw <= 0.001) return null;
    const nx = cx / cw, ny = cy / cw;
    return { x: (nx * 0.5 + 0.5) * this.canvas.clientWidth,
             y: (0.5 - ny * 0.5) * this.canvas.clientHeight,
             dist: cw, visible: nx > -1.3 && nx < 1.3 && ny > -1.3 && ny < 1.3 };
  }
  // meters of world per CSS pixel at a given point (for sizing arrows)
  mppAt(lat, lon, h) {
    const d = vnorm(vsub(ecef(lat, lon, h ?? 0), this.camPos));
    return 2 * d * this.tanHalf / this.canvas.clientHeight;
  }
  // ray through a CSS pixel -> terrain hit {lat, lon, h}; null = sky
  groundRay(px, py) {
    const w = this.canvas.clientWidth, hgt = this.canvas.clientHeight;
    const nx = px / w * 2 - 1, ny = 1 - py / hgt * 2;
    const dir = vunit([
      this.camFwd[0] + nx * this.tanHalf * this.aspect * this.camRight[0] + ny * this.tanHalf * this.camUp[0],
      this.camFwd[1] + nx * this.tanHalf * this.aspect * this.camRight[1] + ny * this.tanHalf * this.camUp[1],
      this.camFwd[2] + nx * this.tanHalf * this.aspect * this.camRight[2] + ny * this.tanHalf * this.camUp[2]]);
    let h = this.terrainAt(this.cam.lat, this.cam.lon) ?? 0;
    let hit = null;
    for (let it = 0; it < 4; it++) {
      const R = GLOBE_R + h;
      const o = this.camPos;
      const b = vdot(o, dir);
      const c = vdot(o, o) - R * R;
      const disc = b * b - c;
      if (disc < 0) return null;
      let t = -b - Math.sqrt(disc);
      if (t < 0) t = -b + Math.sqrt(disc);
      if (t < 0) return null;
      const p = [o[0] + dir[0] * t, o[1] + dir[1] * t, o[2] + dir[2] * t];
      const n = vnorm(p);
      const lat = Math.asin(p[2] / n) * R2D, lon = Math.atan2(p[1], p[0]) * R2D;
      hit = { lat, lon, h };
      const th = this.terrainAt(lat, lon);
      if (th == null) break;
      if (Math.abs(th - h) < 0.5) { hit.h = th; break; }
      h = th;
    }
    return hit;
  }
  screenCenterGround() {
    return this.groundRay(this.canvas.clientWidth / 2, this.canvas.clientHeight / 2);
  }

  /* ---------- render ---------- */
  render() {
    const gl = this.gl;
    this.frame++;
    this.rebuildBudget = 6;
    this._resize();
    this._updateCamera();
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.disable(gl.BLEND);
    gl.disable(gl.CULL_FACE);

    const useH = this.skyMode === "hillaire" && this.atmoOK;
    if (useH) {
      this._buildSkyLUTs();
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    }
    // sky first, no depth
    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    const C = [-this.camPos[0], -this.camPos[1], -this.camPos[2]];
    if (useH) {
      gl.useProgram(this.pSkyH.p);
      const u = this.pSkyH.u;
      gl.uniform3fv(u.uFwd, this.camFwd);
      gl.uniform3fv(u.uRight, this.camRight);
      gl.uniform3fv(u.uUp, this.camUp);
      gl.uniform1f(u.uTanHalf, this.tanHalf);
      gl.uniform1f(u.uAspect, this.aspect);
      gl.uniform3fv(u.uUpW, this.svFrame.up);
      gl.uniform3fv(u.uAzX, this.svFrame.ax);
      gl.uniform3fv(u.uAzY, this.svFrame.ay);
      gl.uniform3fv(u.uSun, this.sunNow);
      gl.uniform1f(u.uCamR, this.camRNow);
      gl.uniform1f(u.uExp, ATMO_EXP);
      gl.uniform3fv(u.uMoon, this.moonNow.dir);
      gl.uniform1f(u.uMoonAng, this.moonNow.ang);
      gl.uniform3fv(u.uMoonZ, this.moonNow.z);
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.moonTex || this.paperTex);
      gl.uniform1i(u.uMoonTex, 3);
      gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, this.tSkyV);
      gl.uniform1i(u.uSkyV, 4);
      gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.tTrans);
      gl.uniform1i(u.uT, 6);
    } else {
      gl.useProgram(this.pSky.p);
      const u = this.pSky.u;
      gl.uniform3fv(u.uFwd, this.camFwd);
      gl.uniform3fv(u.uRight, this.camRight);
      gl.uniform3fv(u.uUp, this.camUp);
      gl.uniform1f(u.uTanHalf, this.tanHalf);
      gl.uniform1f(u.uAspect, this.aspect);
      gl.uniform3fv(u.uEarthC, C);
      gl.uniform1f(u.uR, GLOBE_R);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.skyBuf);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 8, 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // stars: additive over the sky, before terrain (which overdraws them)
    if (this.starCount) {
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.useProgram(this.pStar.p);
      const su = this.pStar.u;
      gl.uniformMatrix4fv(su.uVP, false, this.vp);
      gl.uniform1f(su.uGmst, this.gmstRad(Date.now()));
      gl.uniform1f(su.uDist, 3e7);
      gl.uniform3fv(su.uEarthC, C);
      gl.uniform1f(su.uR, GLOBE_R);
      gl.uniform1f(su.uPxScale, window.devicePixelRatio || 1);
      gl.uniform1f(su.uNight, useH
        ? Math.max(0, Math.min(1, (0.02 - this.sunElevNow) / 0.12)) : 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.starBuf);
      const L = this.starLoc;
      gl.enableVertexAttribArray(L.dir);
      gl.vertexAttribPointer(L.dir, 3, gl.FLOAT, false, 20, 0);
      gl.enableVertexAttribArray(L.mag);
      gl.vertexAttribPointer(L.mag, 1, gl.FLOAT, false, 20, 12);
      gl.enableVertexAttribArray(L.bv);
      gl.vertexAttribPointer(L.bv, 1, gl.FLOAT, false, 20, 16);
      gl.drawArrays(gl.POINTS, 0, this.starCount);
      // the five naked-eye planets ride the same pipeline: equatorial
      // directions and apparent magnitudes recomputed each frame, streamed
      // into a tiny second buffer, rotated by the same GMST
      const pls = this.planetsEq(Date.now());
      if (!this.planetBuf) this.planetBuf = gl.createBuffer();
      const pa = new Float32Array(pls.length * 5);
      pls.forEach((p, i) => pa.set([p.eq[0], p.eq[1], p.eq[2], p.mag, p.bv], i * 5));
      gl.bindBuffer(gl.ARRAY_BUFFER, this.planetBuf);
      gl.bufferData(gl.ARRAY_BUFFER, pa, gl.STREAM_DRAW);
      gl.vertexAttribPointer(L.dir, 3, gl.FLOAT, false, 20, 0);
      gl.vertexAttribPointer(L.mag, 1, gl.FLOAT, false, 20, 12);
      gl.vertexAttribPointer(L.bv, 1, gl.FLOAT, false, 20, 16);
      gl.drawArrays(gl.POINTS, 0, pls.length);
      gl.disableVertexAttribArray(L.dir);
      gl.disableVertexAttribArray(L.mag);
      gl.disableVertexAttribArray(L.bv);
      gl.disable(gl.BLEND);
    }

    // terrain
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.clear(gl.DEPTH_BUFFER_BIT);
    gl.useProgram(this.pTile.p);
    const tu = this.pTile.u;
    gl.uniformMatrix4fv(tu.uVP, false, this.vp);
    gl.uniform1f(tu.uLogF, this.logF);
    const alt = this.camAglNow;
    const fogCol = mixColor([0.905, 0.935, 0.965], [0.55, 0.70, 0.92],
      Math.max(0, Math.min(1, alt / 90000)));
    this._fogColNow = fogCol;
    gl.uniform3fv(tu.uFogCol, fogCol);
    gl.uniform1f(tu.uFogNear, this.fogNear);
    gl.uniform1f(tu.uFogFar, this.fogFar);
    gl.uniform1i(tu.uMode, useH ? 1 : 0);
    if (useH) {
      gl.uniform3fv(tu.uSun, this.sunNow);
      gl.uniform3fv(tu.uEarthC, C);
      gl.uniform1f(tu.uExp, ATMO_EXP);
      gl.uniform3fv(tu.uUpW, this.svFrame.up);
      gl.uniform3fv(tu.uAzX, this.svFrame.ax);
      gl.uniform3fv(tu.uAzY, this.svFrame.ay);
      gl.uniform1f(tu.uCamR, this.camRNow);
      gl.activeTexture(gl.TEXTURE4);
      gl.bindTexture(gl.TEXTURE_2D, this.tSkyV);
      gl.uniform1i(tu.uSkyV, 4);
    }
    gl.uniform1i(tu.uTex, 0);
    gl.activeTexture(gl.TEXTURE0);

    const list = this._traverse();
    this.stats.tiles = list.length;
    this.lastTiles = list.length;
    // load governor: dense scenes (downtown at shallow pitch reached 955
    // tiles) coarsen the refinement target instead of thrashing every
    // cache; quiet scenes ease back toward full detail
    if (list.length > 520) this.refinePx = Math.min(this.refinePx * 1.12, 1024);
    else if (list.length < 320) this.refinePx = Math.max(this.refinePx * 0.94, REFINE_PX);
    list.sort((a, b) => a.z - b.z);          // parents first: children overdraw
    for (const t of list) {
      this._wantTex(t.z, t.x, t.y, t.px);
      const mesh = this._mesh(t.z, t.x, t.y);
      const { tex, uvt } = this._texFor(t.z, t.x, t.y);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform4fv(tu.uUVT, uvt);
      gl.uniform4f(tu.uSolid, 0, 0, 0, 0);
      gl.uniform3f(tu.uOrigin,
        mesh.anchor[0] - this.camPos[0],
        mesh.anchor[1] - this.camPos[1],
        mesh.anchor[2] - this.camPos[2]);
      gl.bindVertexArray(mesh.vao);
      gl.drawElements(gl.TRIANGLES, mesh.n, gl.UNSIGNED_SHORT, 0);
    }
    // polar caps
    for (const cap of this.caps) {
      gl.uniform4fv(tu.uSolid, cap.color);
      gl.uniform4f(tu.uUVT, 1, 1, 0, 0);
      gl.uniform3f(tu.uOrigin,
        cap.anchor[0] - this.camPos[0],
        cap.anchor[1] - this.camPos[1],
        cap.anchor[2] - this.camPos[2]);
      gl.bindVertexArray(cap.vao);
      gl.drawElements(gl.TRIANGLES, cap.n, gl.UNSIGNED_SHORT, 0);
    }
    gl.bindVertexArray(null);

    // buildings
    this._renderBuildings(list);

    // overlay lines: depth-tested against the world, but never writing
    gl.useProgram(this.pLine.p);
    const lu = this.pLine.u;
    gl.uniformMatrix4fv(lu.uVP, false, this.vp);
    gl.uniform1f(lu.uLogF, this.logF);
    gl.uniform2f(lu.uVp, this.canvas.width / 2, this.canvas.height / 2);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for (const s of this.lines.values()) {
      if (s.dirty) this._buildLines(s);
      if (!s.n || !s.anchor) continue;
      gl.uniform3f(lu.uOrigin,
        s.anchor[0] - this.camPos[0],
        s.anchor[1] - this.camPos[1],
        s.anchor[2] - this.camPos[2]);
      gl.uniform1f(lu.uW, s.width * this.dpr);
      gl.uniform4fv(lu.uC, s.color);
      gl.bindVertexArray(s.vao);
      gl.drawElements(gl.TRIANGLES, s.n, gl.UNSIGNED_INT, 0);
    }
    gl.bindVertexArray(null);
    gl.depthMask(true);

    this._pumpImg();
    this._pumpHgt();
    this._placeMarkers();
  }
}

/* ---------- 3D buildings: OpenFreeMap vector tiles, decoded by hand ----------
 * The building layer arrives as Mapbox Vector Tiles (protobuf). A tile is
 * repeated Layers; a Layer has name/features/keys/values/extent; a Feature
 * has tags (key/value index pairs) and a geometry command stream (MoveTo/
 * LineTo/ClosePath with zigzag deltas). Rings with positive shoelace area in
 * raw tile coordinates are exteriors, negative are holes (y grows down, and
 * the spec's CW-exterior convention makes CW read as positive there).
 * Footprints are ear-clipped (holes bridged to the outer ring first) and
 * extruded from local terrain height to terrain + render_height. */
class PbfReader {
  constructor(buf) { this.dv = new DataView(buf); this.pos = 0; this.end = buf.byteLength; }
  varint() {
    let v = 0, shift = 0, b;
    do { b = this.dv.getUint8(this.pos++); v += (b & 0x7f) * Math.pow(2, shift); shift += 7; }
    while (b & 0x80);
    return v;
  }
  skip(wire) {
    if (wire === 0) this.varint();
    else if (wire === 1) this.pos += 8;
    else if (wire === 2) {
      // NOT `this.pos += this.varint()`: += captures the left side BEFORE
      // the call, so the bytes of the length varint itself went uncounted
      // and every skipped field landed short (the source of all the
      // "derailed" layer walks)
      const len = this.varint();
      this.pos += len;
    }
    else if (wire === 5) this.pos += 4;
  }
  // iterate fields of the message ending at `end`; cb(fieldNo, wireType)
  fields(end, cb) {
    while (this.pos < end) {
      const tag = this.varint();
      const wire = tag & 7, no = tag >> 3;
      if (!cb(no, wire)) this.skip(wire);
    }
  }
  bytesEnd() { const len = this.varint(); return this.pos + len; }
  string() { const end = this.bytesEnd();
    const s = new TextDecoder().decode(new Uint8Array(this.dv.buffer, this.pos, end - this.pos));
    this.pos = end; return s; }
}
const zig = v => (v >> 1) ^ -(v & 1);

// -> {polys: [{rings: [[x,y],...][], h, minH}], extent, suspect}
// suspect: a layer walk failed to land exactly on its recorded boundary,
// which means the payload was not a clean vector tile (a mangled response
// once cached an empty building set permanently); callers should retry
function mvtBuildings(buf) {
  const p = new PbfReader(buf);
  let out = null, extent = 4096, suspect = false;
  p.fields(p.end, (no, wire) => {
    if (no !== 3 || wire !== 2) return false;
    const layerEnd = p.bytesEnd();
    // first pass over the layer: find its name before committing to decode
    const save = p.pos;
    let name = null;
    p.fields(Math.min(layerEnd, p.end), (n, w) => {
      if (n === 1 && w === 2) { name = p.string(); return true; }
      return false;
    });
    if (p.pos !== layerEnd) suspect = true;
    p.pos = save;
    if (name !== "building") { p.pos = layerEnd; return true; }
    const feats = [], keys = [], values = [];
    let ext = 4096;
    p.fields(layerEnd, (n, w) => {
      if (n === 1 && w === 2) { p.string(); return true; }
      if (n === 3 && w === 2) { keys.push(p.string()); return true; }
      if (n === 4 && w === 2) {              // Value message
        const vEnd = p.bytesEnd();
        let val = null;
        p.fields(vEnd, (vn, vw) => {
          if (vn === 1 && vw === 2) { val = p.string(); return true; }
          if (vn === 2 && vw === 5) { val = p.dv.getFloat32(p.pos, true); p.pos += 4; return true; }
          if (vn === 3 && vw === 1) { val = p.dv.getFloat64(p.pos, true); p.pos += 8; return true; }
          if ((vn === 4 || vn === 5) && vw === 0) { val = p.varint(); return true; }
          if (vn === 6 && vw === 0) { val = zig(p.varint()); return true; }
          if (vn === 7 && vw === 0) { val = !!p.varint(); return true; }
          return false;
        });
        values.push(val);
        return true;
      }
      if (n === 5 && w === 0) { ext = p.varint(); return true; }
      if (n === 2 && w === 2) {              // Feature
        const fEnd = p.bytesEnd();
        const f = { tags: [], type: 0, geom: null };
        p.fields(fEnd, (fn, fw) => {
          if (fn === 2 && fw === 2) { const e = p.bytesEnd();
            while (p.pos < e) f.tags.push(p.varint()); return true; }
          if (fn === 3 && fw === 0) { f.type = p.varint(); return true; }
          if (fn === 4 && fw === 2) { const e = p.bytesEnd();
            const g = []; while (p.pos < e) g.push(p.varint()); f.geom = g; return true; }
          return false;
        });
        feats.push(f);
        return true;
      }
      return false;
    });
    if (p.pos !== layerEnd) suspect = true;
    extent = ext;
    out = [];
    for (const f of feats) {
      if (f.type !== 3 || !f.geom) continue;
      let h = null, minH = 0;
      for (let i = 0; i + 1 < f.tags.length; i += 2) {
        const k = keys[f.tags[i]], v = values[f.tags[i + 1]];
        if (k === "render_height") h = +v;
        else if (k === "render_min_height") minH = +v;
      }
      if (h == null) h = 8;                  // untagged buildings, as before
      // decode command stream into rings
      const rings = [];
      let ring = null, cx = 0, cy = 0, i = 0;
      const g = f.geom;
      while (i < g.length) {
        const cmd = g[i] & 7, count = g[i] >> 3;
        i++;
        if (cmd === 1) {                     // MoveTo: new ring
          for (let c = 0; c < count; c++) {
            cx += zig(g[i++]); cy += zig(g[i++]);
            ring = [[cx, cy]];
            rings.push(ring);
          }
        } else if (cmd === 2) {
          for (let c = 0; c < count; c++) {
            cx += zig(g[i++]); cy += zig(g[i++]);
            ring.push([cx, cy]);
          }
        } else if (cmd === 7) { /* ClosePath: implicit */ }
      }
      // group rings into polygons by winding (positive shoelace = exterior)
      let cur = null;
      for (const r of rings) {
        if (r.length < 3) continue;
        let a2 = 0;
        for (let k = 0; k < r.length; k++) {
          const [x1, y1] = r[k], [x2, y2] = r[(k + 1) % r.length];
          a2 += x1 * y2 - x2 * y1;
        }
        if (a2 >= 0) { cur = { rings: [r], h, minH }; out.push(cur); }
        else if (cur) cur.rings.push(r);
      }
    }
    p.pos = layerEnd;
    return true;
  });
  return { polys: out || [], extent, suspect };
}

/* ear clipping with holes: bridge each hole to the outer ring at its
 * rightmost vertex, then clip the resulting simple polygon. Coordinates are
 * tile units (y down); the outer ring is normalized to CW-in-y-down (which
 * is CCW in y-up math) before clipping. O(n^2), fine for footprints. */
function triangulateFootprint(outer, holes) {
  const area2 = r => {
    let a = 0;
    for (let i = 0; i < r.length; i++) {
      const [x1, y1] = r[i], [x2, y2] = r[(i + 1) % r.length];
      a += x1 * y2 - x2 * y1;
    }
    return a;
  };
  let poly = outer.slice();
  if (area2(poly) < 0) poly.reverse();
  for (const hole0 of holes) {
    const hole = hole0.slice();
    if (area2(hole) > 0) hole.reverse();       // holes wound opposite
    // rightmost hole vertex, bridged to the nearest outer vertex to its right
    let hi = 0;
    for (let i = 1; i < hole.length; i++) if (hole[i][0] > hole[hi][0]) hi = i;
    let oi = -1, best = Infinity;
    for (let i = 0; i < poly.length; i++) {
      const dx = poly[i][0] - hole[hi][0];
      if (dx < 0) continue;
      const d = dx * dx + (poly[i][1] - hole[hi][1]) ** 2;
      if (d < best) { best = d; oi = i; }
    }
    if (oi < 0) oi = 0;
    const spliced = poly.slice(0, oi + 1)
      .concat(hole.slice(hi), hole.slice(0, hi + 1), poly.slice(oi));
    poly = spliced;
  }
  // ear clip
  const n0 = poly.length;
  const idx = poly.map((_, i) => i);
  const tris = [];
  const cross = (a, b, c) => (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
  const inTri = (p, a, b, c) =>
    cross(a, b, p) >= -1e-9 && cross(b, c, p) >= -1e-9 && cross(c, a, p) >= -1e-9;
  let guard = 0;
  while (idx.length > 3 && guard++ < n0 * n0) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const ia = idx[(i + idx.length - 1) % idx.length], ib = idx[i],
            ic = idx[(i + 1) % idx.length];
      const a = poly[ia], b = poly[ib], c = poly[ic];
      if (cross(a, b, c) <= 1e-9) continue;    // reflex or degenerate
      let ok = true;
      for (const j of idx) {
        if (j === ia || j === ib || j === ic) continue;
        if (inTri(poly[j], a, b, c)) { ok = false; break; }
      }
      if (!ok) continue;
      tris.push(ia, ib, ic);
      idx.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) break;                       // degenerate input: stop clean
  }
  if (idx.length === 3) tris.push(idx[0], idx[1], idx[2]);
  return { verts: poly, tris };
}

const BLDG_Z = 14;            // OpenFreeMap building layer, full detail
const BLDG_DIST = 9000;       // m: fetch/draw radius around the camera
const BLDG_CAP = 90;          // resident building-tile meshes

Globe.prototype._bldgInit = function () {
  if (this.bldg) return;
  this.bldg = { meshes: new Map(), inflight: new Set(), pending: new Map(),
                tpl: null, tplLoading: false };
  this.buildingsRender = list => this._renderBuildings(list);
};
Globe.prototype._bldgFetchTpl = function () {
  const B = this.bldg;
  if (B.tpl || B.tplLoading) return;
  B.tplLoading = true;
  fetch("https://tiles.openfreemap.org/planet").then(r => r.json()).then(j => {
    B.tpl = j.tiles[0];
  }).catch(() => { B.tplLoading = false; });
};
Globe.prototype._bldgBuild = function (z, x, y, data) {
  const gl = this.gl;
  const geo = this._tileGeo(z, x, y);
  const anchor = ecef(geo.latM, geo.lonM, 0);
  const up = vunit(anchor);
  const sun = vunit([up[0] * 1.2 + 0.6, up[1] * 1.2 + 0.35, up[2] * 1.2 + 0.5]);
  const { polys, extent } = data;
  const vs = [], is = [];
  const toLL = (px, py) => [ty2lat(y + py / extent, z), tx2lon(x + px / extent, z)];
  for (const poly of polys) {
    const outer = poly.rings[0];
    const [bLat, bLon] = toLL(outer[0][0], outer[0][1]);
    const base = this.terrainAt(bLat, bLon) ?? 0;
    const top = base + Math.max(poly.h, 2), bot = base + poly.minH;
    // roof
    const { verts, tris } = triangulateFootprint(outer, poly.rings.slice(1));
    const roofBase = vs.length / 6;
    for (const [px, py] of verts) {
      const [la, lo] = toLL(px, py);
      const p = vsub(ecef(la, lo, top), anchor);
      vs.push(p[0], p[1], p[2], up[0], up[1], up[2]);
    }
    for (const t of tris) is.push(roofBase + t);
    // walls, all rings (holes give courtyard walls)
    for (const ring of poly.rings) {
      for (let i = 0; i < ring.length; i++) {
        const [x1, y1] = ring[i], [x2, y2] = ring[(i + 1) % ring.length];
        if (x1 === x2 && y1 === y2) continue;
        const [laA, loA] = toLL(x1, y1), [laB, loB] = toLL(x2, y2);
        const aT = vsub(ecef(laA, loA, top), anchor), aB = vsub(ecef(laA, loA, bot), anchor);
        const bT = vsub(ecef(laB, loB, top), anchor), bB = vsub(ecef(laB, loB, bot), anchor);
        // exterior rings arrive clockwise in the geographic frame, so the
        // outward face normal is up x edge (holes, wound opposite, flip
        // automatically toward their courtyard)
        const nrm = vunit(vcross(up, vsub(bT, aT)));
        const b0 = vs.length / 6;
        for (const p of [aB, bB, bT, aT])
          vs.push(p[0], p[1], p[2], nrm[0], nrm[1], nrm[2]);
        is.push(b0, b0 + 1, b0 + 2, b0, b0 + 2, b0 + 3);
      }
    }
  }
  const vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const vb = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vs), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
  gl.enableVertexAttribArray(1);
  gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
  const ib = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint32Array(is), gl.STATIC_DRAW);
  gl.bindVertexArray(null);
  return { vao, vb, ib, n: is.length, anchor, sun, geo, frame: this.frame,
           hgtStamp: this._hgtStampFor(z, x, y), data };
};
Globe.prototype._renderBuildings = function (list) {
  if (this.camAglNow > 12000) return;          // invisible from up there
  this._bldgInit();
  const B = this.bldg;
  // which z14 building tiles are close enough to matter this frame
  const want = new Set();
  for (const t of list) {
    if (t.z < BLDG_Z || t.dist > BLDG_DIST) continue;
    const dz = t.z - BLDG_Z;
    want.add(tkey(BLDG_Z, t.x >> dz, t.y >> dz));
  }
  if (!want.size) return;
  this._bldgFetchTpl();
  if (B.tpl) {
    for (const key of want) {
      const have = B.meshes.get(key);
      if (have && !(have.retryAt && have.tries < 4 && performance.now() > have.retryAt)) continue;
      if (B.inflight.has(key)) continue;
      if (B.inflight.size >= 4) break;
      B.inflight.add(key);
      const [z, x, y] = key.split("/").map(Number);
      const url = B.tpl.replace("{z}", z).replace("{x}", x).replace("{y}", y);
      const prior = B.meshes.get(key);
      const tries = (prior && prior.tries) || 0;
      fetch(url).then(r => r.ok ? r.arrayBuffer() : Promise.reject(r.status))
        .then(buf => {
          const data = mvtBuildings(buf);
          if (data.suspect && tries < 3)   // mangled payload: retry, don't cache
            B.meshes.set(key, { n: 0, frame: this.frame, tries: tries + 1,
                                retryAt: performance.now() + 15000 });
          else
            B.pending.set(key, data);   // mesh built later, one per frame:
                                        // a dense downtown tile costs
                                        // 13-48 ms and several completing
                                        // at once froze the frame
        })
        .catch(() => {
          B.meshes.set(key, { n: 0, frame: this.frame, tries: tries + 1,
                              retryAt: performance.now() + 15000 });
        })
        .finally(() => B.inflight.delete(key));
    }
  }
  // build at most one pending mesh per frame
  let builtThisFrame = false;
  if (B.pending.size) {
    const [key, data] = B.pending.entries().next().value;
    B.pending.delete(key);
    const [z, x, y] = key.split("/").map(Number);
    B.meshes.set(key, this._bldgBuild(z, x, y, data));
    this.stats.buildings = B.meshes.size;
    builtThisFrame = true;
  }
  // draw
  const gl = this.gl;
  gl.useProgram(this.pBldg.p);
  const u = this.pBldg.u;
  gl.uniformMatrix4fv(u.uVP, false, this.vp);
  gl.uniform1f(u.uLogF, this.logF);
  gl.uniform3fv(u.uFogCol, this._fogColNow);
  gl.uniform1f(u.uFogNear, this.fogNear);
  gl.uniform1f(u.uFogFar, this.fogFar);
  const useH = this.skyMode === "hillaire" && this.atmoOK && this.sunNow;
  gl.uniform1i(u.uMode, useH ? 1 : 0);
  if (useH) {
    gl.uniform3fv(u.uEarthC,
      [-this.camPos[0], -this.camPos[1], -this.camPos[2]]);
    gl.uniform1f(u.uExp, ATMO_EXP);
    gl.uniform3fv(u.uUpW, this.svFrame.up);
    gl.uniform3fv(u.uAzX, this.svFrame.ax);
    gl.uniform3fv(u.uAzY, this.svFrame.ay);
    gl.uniform1f(u.uCamR, this.camRNow);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.tSkyV);
    gl.uniform1i(u.uSkyV, 4);
  }
  for (const key of want) {
    const m = B.meshes.get(key);
    if (!m || !m.n) continue;
    m.frame = this.frame;
    // terrain arrived since this mesh was built: rebase the buildings
    // (same one-build-per-frame budget as fresh builds)
    if (!builtThisFrame) {
      const [z, x, y] = key.split("/").map(Number);
      if (this._hgtStampFor(z, x, y) !== m.hgtStamp) {
        gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.vb); gl.deleteBuffer(m.ib);
        B.meshes.set(key, this._bldgBuild(z, x, y, m.data));
        builtThisFrame = true;
        continue;
      }
    }
    gl.uniform3fv(u.uSun, useH ? this.sunNow : m.sun);
    gl.uniform3f(u.uOrigin,
      m.anchor[0] - this.camPos[0],
      m.anchor[1] - this.camPos[1],
      m.anchor[2] - this.camPos[2]);
    gl.bindVertexArray(m.vao);
    gl.drawElements(gl.TRIANGLES, m.n, gl.UNSIGNED_INT, 0);
  }
  gl.bindVertexArray(null);
  // LRU
  if (B.meshes.size > BLDG_CAP) {
    const ent = [...B.meshes.entries()].sort((a, b) => a[1].frame - b[1].frame);
    for (let i = 0; i < ent.length - BLDG_CAP + 15; i++) {
      const m = ent[i][1];
      if (m.vao) { gl.deleteVertexArray(m.vao); gl.deleteBuffer(m.vb); gl.deleteBuffer(m.ib); }
      B.meshes.delete(ent[i][0]);
    }
  }
};

function mat4mul(a, b) {     // column-major, result = a * b
  const o = new Array(16);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1]
                   + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}
function mixColor(a, b, t) {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
}
