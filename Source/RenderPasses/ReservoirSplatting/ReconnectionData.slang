/***************************************************************************
 # Copyright (c) 2015-25, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/

import Scene.Raytracing;
import Scene.Intersection;
import Utils.Sampling.SampleGenerator;
import Utils.Debug.PixelDebug;

// This struct is used to package the data needed when shifting.
struct ShiftedPathData
{
    // TODO: technically, the scatter / gather shift only ever needs one of the two here,
    // but both are stored here for simplicity.
    HitInfo primaryHit;
    float2 fractionalPixel;

    float2 lensSample;  // This is sometimes used for Area ReSTIR DoF.
    float3 firstRayDir; // This is sometimes used for environment maps.

    // Track all the Jacobian footprints used for all the shifts.
    float subPixelJacobian; // The reprojection (splatting) shift changes the subpixel location.
    float lensVertexJacobian; // The primary hit reconnection shift changes the lens vertex.
    float secondaryPathJacobian; // The path reconnection shift modifies the path density.

    // Shifted radiance value.
    float3 radiance;

    __init()
    {
        this.primaryHit = HitInfo();
        this.fractionalPixel = float2(-1.0f, -1.0f);

        this.lensSample = float2(0.0f, 0.0f);
        this.firstRayDir = float3(0.0f, 0.0f, 0.0f);

        this.subPixelJacobian = 1.0f;
        this.lensVertexJacobian = 1.0f;
        this.secondaryPathJacobian = 1.0f;

        this.radiance = float3(0.0f, 0.0f, 0.0f);
    }

    __init(HitInfo initialPrimaryHit, float2 initialFractionalPixel, float2 initialLensSample, float3 initialFirstRayDir, float initialSubPixelJacobian, float initialLensVertexJacobian, float initialSecondaryPathJacobian, float3 initialRadiance)
    {
        this.primaryHit = initialPrimaryHit;
        this.fractionalPixel = initialFractionalPixel;

        this.lensSample = initialLensSample;
        this.firstRayDir = initialFirstRayDir,

        this.subPixelJacobian = initialSubPixelJacobian;
        this.lensVertexJacobian = initialLensVertexJacobian;
        this.secondaryPathJacobian = initialSecondaryPathJacobian;

        this.radiance = initialRadiance;
    }
};

// The reconnection vertex is always the second hit.
// The three options are:
//   1. The primary hit is a light (reconnection is impossible).
//   2. The second hit is a light (i.e. the path length is 2).
//   3. The second hit is a normal vertex in the path.
struct ReconnectionData
{
    // Some camera / film parameters used generally.
    float2 subPixel;
    float2 lensSample;
    float time;

    // The only vertex that contributes actual emission is the final vertex,
    // i.e. the path length is equivalent to the index of the light vertex.
    uint pathLength;

    // Information about the first vertex.
    HitInfo firstHit; // Hit info of the first vertex.
    uint firstBSDFComponentType; // The BSDF component mask of the first hit.
    float3 firstWi; // The incoming ray direction of the first hit.

    // Information about the second vertex.
    HitInfo secondHit; // Hit info of the second vertex.
    uint secondBSDFComponentType; // The BSDF component mask of the second hit.
    float3 secondWo;             // The outgoing ray direction after the second hit.

    bool transmissionEvent; // Track if the first -> second vertex is a transmission event.

    // If the second or third vertex is a light, then we need to handle things slightly differently.
    bool lightIsNEE; // If the sampled light was sampled by NEE or not.
    bool lightIsDistant; // If the sampled light was distant or not.
    float lightPdf; // Probability of selecting the light.
    // When the light is the second vertex, this is in area measure. If it's the third vertex, then it's the solid angle measure.
    // But the third vertex / solid angle measure is only really useful for MIS since they cancel out in the Jacobian.

    float subPixelJacobian;       // For scatter-based shifting, this is needed to re-parametrize from primary hit world-space to subpixel coordinate.
    float lensVertexJacobian;       // For gather-based primary hit reconnection shift, this is needed to re-parametrize from primary hit world-space to lens position.
    float secondaryPathJacobian;    // For the reconnection shift, this is needed to re-parametrize from second hit PSS to solid angle.
    float3 irradiance;      // This gets used for different things depending on which case occurs.
    float3 earlyThroughput; // Throughput past the first/second vertex to reconstruct the total irradiance.

    __init()
    {
        this.subPixel = float2(0.5f, 0.5f);
        this.lensSample = float2(0.0f, 0.0f);
        this.time = 0.0f;

        this.pathLength = 0;

        this.firstHit = HitInfo();
        this.firstBSDFComponentType = 0;
        this.firstWi = float3(0.0f, 0.0f, 0.0f);

        this.secondHit = HitInfo();
        this.secondBSDFComponentType = 0;
        this.secondWo = float3(0.0f, 0.0f, 0.0f);

        this.transmissionEvent = false;

        this.lightIsNEE = false;
        this.lightIsDistant = false;
        this.lightPdf = 0.0f;

        this.subPixelJacobian = 1.0f;
        this.lensVertexJacobian = 1.0f;
        this.secondaryPathJacobian = 1.0f;
        this.irradiance = float3(0.0f, 0.0f, 0.0f);
        this.earlyThroughput = float3(1.0f, 1.0f, 1.0f);
    }

    [mutating]
    void update(ShiftedPathData shiftedPath)
    {
        this.firstHit = shiftedPath.primaryHit;
        this.subPixel = shiftedPath.fractionalPixel - floor(shiftedPath.fractionalPixel);
        this.lensSample = shiftedPath.lensSample;
        this.firstWi = -shiftedPath.firstRayDir;

        this.subPixelJacobian = shiftedPath.subPixelJacobian;
        this.lensVertexJacobian = shiftedPath.lensVertexJacobian;
        this.secondaryPathJacobian = shiftedPath.secondaryPathJacobian;
    }
};
