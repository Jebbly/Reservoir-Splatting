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

import ShiftOptions;
import ShiftMapping;

struct RobustReuseOptimization
{
    PathTracerParams params;

    StructuredBuffer<PathReservoir> prevReservoirs;
    StructuredBuffer<ReconnectionData> prevReconnectionData;

    // XYZ is the shifted integrand, W is the shifted destination Jacobian.
    RWStructuredBuffer<ShiftedPathData> shiftedPaths;

    static int2 kOffsets[8] = {
        int2(-1, -1), int2(0, -1), int2(1, -1),
        int2(-1, 0),  /* no-op */  int2(1, 0),
        int2(-1, 1),  int2(0, 1),  int2(1, 1)
    };

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        print("======= Robust Reuse Optimization =======", pixel);

        // TODO: use a better seed here.
        SampleGenerator sg = SampleGenerator(pixel, (params.useFixedSeed > 0) ? params.fixedSeed + 1: kNumRenderPasses * (params.frameCount + params.seed) + 1);

        uint32_t reservoirIdx = params.linearizePixel(pixel);

        // At this point we can assume that the (previous) central reservoir exists and is valid.
        PathReservoir prevReservoir = prevReservoirs[reservoirIdx];
        ReconnectionData prevReconnection = prevReconnectionData[reservoirIdx];

        for (uint offsetIndex = 0; offsetIndex < 8; offsetIndex++)
        {
            // Now we compute a shift into a neighbor.
            int2 neighborOffset = kOffsets[offsetIndex];
            int2 neighborPixel = int2(pixel) + neighborOffset;
            bool neighborIsValid = params.validateFloatingPixelBounds(float2(neighborPixel));
            bool canShift = any(prevReservoir.integrand > 0.0f);

            ShiftedPathData shiftedPath = (neighborIsValid && canShift) ? gatherLensVertexCopyShift(sg, prevReconnection, prevReconnection.time + gCameraManager.artificialFrameTime, float2(neighborPixel) + prevReservoir.subPixel, prevReconnection.lensSample) 
                                                                        : ShiftedPathData();

            shiftedPaths[8 * reservoirIdx + offsetIndex] = shiftedPath;
        }
    }
}

cbuffer CB
{
    RobustReuseOptimization gRobustReuseOptimization;
}

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
    gRobustReuseOptimization.run(pixel);
}
