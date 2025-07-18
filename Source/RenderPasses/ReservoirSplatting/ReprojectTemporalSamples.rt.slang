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
 
#include "Utils/Math/MathConstants.slangh"
#include "Scene/SceneDefines.slangh"
#include "RaytracingCommon.slang"

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;

import LoadShadingData;
import PathTracer;
import ReconnectionData;
import Reservoir;

import Params;
import ShiftOptions;

ParameterBlock<PathTracer> gPathTracer;

struct ReprojectTemporalSamples
{
    PathTracerParams params;

    static const uint kCounterIndexDataCount = 0;
    static const uint kCounterIndexPrefixSum = 1;
    RWStructuredBuffer<uint32_t> globalCounters;
    RWStructuredBuffer<uint32_t> cellCounters;
    RWStructuredBuffer<uint2> reservoirIndices;
    RWStructuredBuffer<uint2> scatteredReservoirs;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> prevReconnectionData;

    /**
     * Scatter a reservoir from the previous frame into the current frame.
     * @param[in] pixel Current pixel coordinate.
     */
    void run(const uint2 pixel)
    {
        if (!params.validateIntegerPixelBounds(pixel)) return;
        printSetPixel(pixel);
        print("======= Reproject Temporal Samples =======", pixel);

        uint32_t currIndex = params.linearizePixel(pixel);
        PathReservoir prevReservoir = prevReservoirs[currIndex];
        if (all(prevReservoir.integrand == 0.0f))
        {
            print("Previous reservoir is empty", false);
            return;
        }

        ReconnectionData prevReconnection = prevReconnectionData[currIndex];

        // Get the first hit position of the old reservoir.
        Camera prevCamera = gScene.camera;
        gCameraManager.orientCamera(prevCamera, params.frameCount, gCameraManager.artificialFrameTime + prevReconnection.time);

        // Find where the hit corresponds to in the new (current) frame.
        Camera currCamera = gScene.camera;
        gCameraManager.orientCamera(currCamera, params.frameCount, prevReconnection.time);

        // A new ray must be constructed to check for visibility.
        Ray newRay;

        // Get the origin offset from the lens sample, which is always (0, 0) for a pinhole.
        float2 lensLocal = currCamera.data.apertureRadius * prevReconnection.lensSample;
        print("Lens Local", lensLocal);
        float3 lensWorld = lensLocal.x * normalize(currCamera.data.cameraU) + lensLocal.y * normalize(currCamera.data.cameraV);
        newRay.origin = currCamera.data.posW + lensWorld;

        // Get the ray direction depending on the type of sample (primary hit or environment map).
        if (prevReconnection.firstHit.isValid())
        {
            // If the primary hit exists, we want to get the location.
            ShadingData firstSd = loadShadingData(prevReconnection.firstHit, prevCamera.data.posW, float3(0.0f, 0.0f, 0.0f));
            float3 x1 = firstSd.posW;

            newRay.dir = normalize(x1 - newRay.origin);
            newRay.tMin = 0.001f * length(x1 - newRay.origin);
            newRay.tMax = 0.999f * length(x1 - newRay.origin);
        }
        else
        {
            // Even if the first hit isn't valid, we can still reproject if the previous sample was an environment map
            // by matching ray directions from the previous frame into the current frame.
            if (prevReconnection.lightIsDistant)
            {
                newRay.dir = -prevReconnection.firstWi;
                newRay.tMin = 0.001f;
                newRay.tMax = kRayTMax;
            }
            else
            {
                // Otherwise, the hit was just invalid, so we can't reproject.
                return;
            }
        }

        // Convert the new ray to camera space.
        float3 camDir = float3(dot(normalize(currCamera.data.cameraU), newRay.dir),
                               dot(normalize(currCamera.data.cameraV), newRay.dir),
                               dot(normalize(currCamera.data.cameraW), newRay.dir));


        // If the new position is behind the camera, don't scatter.
        if (camDir.z <= 0.001f)
        {
            print("Point is behind camera", camDir.z);
            return;
        }

        // Cast a visibility ray to the hit point, to check if the scattered reservoir hasn't been covered yet.
        if (!gPathTracer.traceVisibilityRay(newRay))
        {
            print("Scattered reservoir is no longer visible", false);
            return;
        }

        // float tanFovY = currCamera.data.focalLength == 0.0f ? 0.0f : currCamera.data.frameHeight / currCamera.data.focalLength;
        // d *= tanFovY;

        // float2 offset = float2(dot(normalizedRayDir, normalize(currCamera.data.cameraU)), -dot(normalizedRayDir, normalize(currCamera.data.cameraV)));
        //  float2 ndc = float2(0.5f, 0.5f) + offset / float2(d * currCamera.data.aspectRatio, d);
        float2 film = lensLocal + length(currCamera.data.cameraW) * (camDir.xy / camDir.z);
        float2 ndc = film / float2(length(currCamera.data.cameraU), length(currCamera.data.cameraV));
        float2 newFractionalPixel = (float2(0.5f, -0.5f) * ndc + float2(0.5f, 0.5f)) * params.frameDim;

        // If the new position doesn't lie in the camera frame, don't scatter.
        if (!params.validateFloatingPixelBounds(newFractionalPixel))
        {
            print("New scattered position falls outside of camera", newFractionalPixel);
            return;
        }

        uint2 newPixel = uint2(floor(newFractionalPixel));
        uint linearizedIndex = params.linearizePixel(newPixel);

        print("Current Pixel", pixel); 
        print("New Pixel", newPixel);

        uint index;
        InterlockedAdd(globalCounters[kCounterIndexDataCount], 1, index);

        uint cellIndex;
        InterlockedAdd(cellCounters[linearizedIndex], 1, cellIndex);

        reservoirIndices[index] = uint2(linearizedIndex, cellIndex);
        scatteredReservoirs[index] = pixel;
    }
};

cbuffer CB
{
    ReprojectTemporalSamples gReprojectTemporalSamples;
};

//
// Shader entry point for ray generation shader.
//

[shader("raygeneration")]
void rayGen()
{
    uint2 pixel = DispatchRaysIndex().xy;
    uint2 frameDim = DispatchRaysDimensions().xy;
    if (all(pixel >= frameDim)) return;

    printSetPixel(pixel);
    gReprojectTemporalSamples.run(pixel);
}
