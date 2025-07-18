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

struct ScatterTemporalResampling
{
    bool useConfidenceWeights;

    PathTracerParams params;

    Texture2D<float2> motionVectors;

    StructuredBuffer<uint32_t> cellCounters;    // Number of reservoirs that were scattered into a given cell.
    StructuredBuffer<uint> cellOffsets;         // Where the initial cell offset begins.
    StructuredBuffer<uint2> sortedReservoirs;   // Linearized buffer of scattered reservoirs.

    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PathReservoir> currReservoirs;

    StructuredBuffer<ReconnectionData> prevReconnectionData;
    RWStructuredBuffer<ReconnectionData> currReconnectionData;

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        print("======= Scatter Temporal Resampling =======", pixel);

        // TODO: use a better seed here.
        SampleGenerator sg = SampleGenerator(pixel, (params.useFixedSeed > 0) ? params.fixedSeed + 5 : kNumRenderPasses * (params.frameCount + params.seed) + 5);

        PathReservoir dstReservoir = PathReservoir();
        ReconnectionData dstReconnectionData = ReconnectionData();

        uint32_t reservoirIdx = params.linearizePixel(pixel);

        // We first consider the current reservoir (the initial sample).
        PathReservoir currReservoir = currReservoirs[reservoirIdx];
        ReconnectionData currReconnection = currReconnectionData[reservoirIdx];
        float currSampleMIS = 1.0f;
        if (any(currReservoir.integrand > 0.0f))
        {
            float m1 = luminance(currReservoir.integrand) * (useConfidenceWeights ? currReservoir.confidence : 1.0f);

            // Shift the current sample to the previous frame domain.
            float m2 = 0.0f;
            ShiftedPathData shiftedCurr = scatterReprojectionShift(sg, currReconnection, currReconnection.time + gCameraManager.artificialFrameTime, currReconnection.firstHit, currReconnection.lensSample);
            float shiftedJacobian = (shiftedCurr.subPixelJacobian * shiftedCurr.secondaryPathJacobian) / (currReconnection.subPixelJacobian * currReconnection.secondaryPathJacobian);
            m2 = luminance(shiftedCurr.radiance) * shiftedJacobian;
            m2 = isnan(m2) ? 0.0f : m2;

            int2 scatteredPixel = int2(floor(shiftedCurr.fractionalPixel));
            if (params.validateIntegerPixelBounds(scatteredPixel))
            {
                PathReservoir prevReservoir = prevReservoirs[params.linearizePixel(scatteredPixel)];
                m2 *= useConfidenceWeights ? prevReservoir.confidence : 1.0f;
            }
            else
            {
                m2 = 0.0f;
            }

            currSampleMIS = (m1 + m2 == 0.0f) ? 0.0f : m1 / (m1 + m2);
        }

        bool currSelected = dstReservoir.addSampleFromReservoir(sg, currSampleMIS, currReservoir.integrand, 1.0f, currReservoir);
        dstReconnectionData = currSelected ? currReconnection : dstReconnectionData;

        // Keep track of the confidence after the current sample.
        float newConfidence = dstReservoir.confidence;

        // Next, consider the scattered reservoirs from the previous frame.
        uint32_t numReservoirs = cellCounters[reservoirIdx];
        uint32_t cellOffset = cellOffsets[reservoirIdx];
        for (uint32_t i = 0; i < numReservoirs; i++)
        {
            uint2 scatteredPixel = sortedReservoirs[cellOffset + i];
            // Anything that was scattered should already be valid.
            uint32_t scatteredIdx = params.linearizePixel(scatteredPixel);
            PathReservoir prevReservoir = prevReservoirs[scatteredIdx];
            ReconnectionData prevReconnection = prevReconnectionData[scatteredIdx];

            // Shift the previous sample to the current domain.
            float prevSampleMIS = 0.0f;
            float3 prevPHat = float3(0.0f, 0.0f, 0.0f);
            float shiftedJacobian = 1.0f;
            if (any(prevReservoir.integrand > 0.0f))
            {
                ShiftedPathData shiftedPrev = scatterReprojectionShift(sg, prevReconnection, prevReconnection.time, prevReconnection.firstHit, prevReconnection.lensSample);
                shiftedJacobian = (shiftedPrev.subPixelJacobian * shiftedPrev.secondaryPathJacobian) / (prevReconnection.subPixelJacobian * prevReconnection.secondaryPathJacobian);
                float m1 = luminance(shiftedPrev.radiance) * shiftedJacobian * (useConfidenceWeights ? currReservoir.confidence : 1.0f);
                prevPHat = isnan(m1) ? float3(0.0f, 0.0f, 0.0f) : shiftedPrev.radiance;
                shiftedJacobian = isnan(m1) ? 1.0f : shiftedJacobian; // W_{y_i} = W_{x_i} * |T(x)|

                // Update the relevant data.
                prevReconnection.update(shiftedPrev);

                // The scattered reservoir will also be at a new subpixel location, compared to where it originally started.
                prevReservoir.setSubPixel(prevReconnection.subPixel);
                float m2 = luminance(prevReservoir.integrand) * (useConfidenceWeights ? prevReservoir.confidence : 1.0f);

                prevSampleMIS = ((m1 + m2) == 0.0f) ? 0.0f : m2 / (m1 + m2);
            }
            
            print("Previous Sample MIS", prevSampleMIS);
            bool prevSelected = dstReservoir.addSampleFromReservoir(sg, prevSampleMIS, prevPHat, shiftedJacobian, prevReservoir);
            dstReconnectionData = prevSelected ? prevReconnection : dstReconnectionData;
        }

        // We use the motion vector to compute the new confidence weight (same as Area ReSTIR).
        float2 motionVector = motionVectors[pixel];
        motionVector = (length(motionVector) < 1e-06) ? float2(0.0f, 0.0f) : motionVector;
        float2 prevPixel = float2(pixel) + motionVector * float2(params.frameDim);
        int2 prevPixelTopLeft = int2(floor(prevPixel));
        float2 fractionalCoord = saturate(prevPixel - float2(prevPixelTopLeft));
        for (int x = 0; x < 2; x++)
        {
            for (int y = 0; y < 2; y++)
            {
                float2 offset = float2(x, y);
                uint2 neighborPixel = prevPixelTopLeft + uint2(offset);
                if (!params.validateIntegerPixelBounds(neighborPixel)) continue;

                uint32_t neighborIndex = params.linearizePixel(neighborPixel);
                PathReservoir neighborReservoir = prevReservoirs[neighborIndex];
                float bilinearWeight = lerp(1 - x, x, fractionalCoord.x) * lerp(1 - y, y, fractionalCoord.y);
                newConfidence += bilinearWeight * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);
            }
        }
        dstReservoir.confidence = min(PathReservoir.confidenceCap, newConfidence);

        currReservoirs[reservoirIdx] = dstReservoir;
        currReconnectionData[reservoirIdx] = dstReconnectionData;
    }
}

cbuffer CB
{
    ScatterTemporalResampling gScatterTemporalResampling;
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
    gScatterTemporalResampling.run(pixel);
}
