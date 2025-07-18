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
import ShiftMapping;

struct SpatialResampling
{
    PathTracerParams params;
    bool useConfidenceWeights;

    uint32_t iteration;
    uint32_t neighborCount;
    Texture1D<float2> neighborOffsets;

    float gatherRadius;

    StructuredBuffer<ReconnectionData> prevReconnectionData;
    StructuredBuffer<PathReservoir> prevReservoirs;

    RWStructuredBuffer<ReconnectionData> currReconnectionData;
    RWStructuredBuffer<PathReservoir> currReservoirs;

    Texture2D<PackedHitInfo> vbuffer;

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        print("======= Spatial Resampling =======", pixel);

        // TODO: use a better seed here (this is just a random number).
        SampleGenerator sg = SampleGenerator(pixel, (params.useFixedSeed > 0) ? params.fixedSeed + 7 : kNumRenderPasses * (params.frameCount + params.seed) + 1337 * iteration + 7);

        uint32_t reservoirIdx = params.linearizePixel(pixel);

        const bool hasDepthOfField = (gScene.camera.data.apertureRadius > 0.0f);
        float2 depthOfFieldProbs = float2(1.0f, 0.0f);
        if (hasDepthOfField)
        {
            const HitInfo hit = HitInfo(vbuffer[pixel]);

            float circleOfConfusion = 0.0f;
            if (hit.isValid())
            {
                ShadingData sd = loadShadingData(hit, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));
                circleOfConfusion = computePrimaryHitCircleOfConfusion(sd.posW);
            }
            else
            {
                circleOfConfusion = computeEnvMapCircleOfConfusion();
            }

            depthOfFieldProbs = computeDepthOfFieldGatherShiftProbabilities(circleOfConfusion);
        }

        const uint numLensVertexCopyShifts = uint(round(depthOfFieldProbs.x * neighborCount));

        // We first consider the central reservoir (this is put in the "previous" buffer)
        // in order to avoid any R/W race conditions.
        PathReservoir centralReservoir = prevReservoirs[reservoirIdx];
        ReconnectionData centralReconnection = prevReconnectionData[reservoirIdx];
        float centralSampleMIS = 1.0f;
        float centralWeight = luminance(centralReservoir.integrand) / float(neighborCount) * (useConfidenceWeights ? centralReservoir.confidence : 1.0f);

        PathReservoir dstReservoir = PathReservoir();
        ReconnectionData dstReconnectionData = ReconnectionData();

        int validNeighbors = 0;
        uint32_t startIdx = sampleNext1D(sg) * kSpatialNeighborSamples;
        for (uint32_t i = 0; i < neighborCount; i++)
        {
            // Lens vertex copy or primary hit reconnection, depending on which index we're at.
            bool lensVertexCopyShift = (i < numLensVertexCopyShifts);

            // Get the neighbor pixel.
            uint32_t neighborOffsetIdx = (startIdx + i) % kSpatialNeighborSamples;
            int2 neighborPixel = int2(round(float2(pixel) + gatherRadius * neighborOffsets[neighborOffsetIdx]));
            if (!params.validateIntegerPixelBounds(neighborPixel)) continue;
            validNeighbors += 1;

            uint32_t neighborIdx = params.linearizePixel(neighborPixel);
            PathReservoir neighborReservoir = prevReservoirs[neighborIdx];
            ReconnectionData neighborReconnection = prevReconnectionData[neighborIdx];

            // Shift the central sample to the neighbor domain.
            float m1 = 0.0f;
            if (any(centralReservoir.integrand > 0.0f))
            {
                // Shift central sample to neighbor domain.
                float2 fractionalPixel = float2(neighborPixel) + centralReservoir.subPixel;
                ShiftedPathData shiftedCentral = lensVertexCopyShift ? gatherLensVertexCopyShift(sg, centralReconnection, centralReconnection.time, fractionalPixel, centralReconnection.lensSample)
                                                                     : gatherPrimaryHitReconnectionShift( sg, centralReconnection, centralReconnection.time, fractionalPixel, centralReconnection.firstHit);
                
                // The lens vertex copy does not include the lens vertex Jacobian.
                float shiftedJacobian = (lensVertexCopyShift ? 1.0f : (shiftedCentral.lensVertexJacobian / centralReconnection.lensVertexJacobian)) * 
                                                                (shiftedCentral.secondaryPathJacobian / centralReconnection.secondaryPathJacobian);

                m1 = luminance(shiftedCentral.radiance) * shiftedJacobian * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);
                m1 = isnan(m1) ? 0.0f : m1;
            }
            centralSampleMIS += 1.0f;
            centralSampleMIS -= (m1 + centralWeight > 0.0f) ? m1 / (m1 + centralWeight) : 0.0f;

            // Shift the neighbor sample to the central domain.
            float neighborSampleMIS = 0.0f;
            float3 neighborPHat = float3(0.0f, 0.0f, 0.0f);
            float shiftedJacobian = 1.0f;
            if (any(neighborReservoir.integrand > 0.0f))
            {
                // Shift neighbor pixel to central domain.
                float2 fractionalPixel = float2(pixel) + neighborReservoir.subPixel;
                ShiftedPathData shiftedNeighbor = lensVertexCopyShift ? gatherLensVertexCopyShift(sg, neighborReconnection, neighborReconnection.time, fractionalPixel, neighborReconnection.lensSample)
                                                                      : gatherPrimaryHitReconnectionShift( sg, neighborReconnection, neighborReconnection.time, fractionalPixel, neighborReconnection.firstHit);

                // The lens vertex copy does not include the lens vertex Jacobian.
                shiftedJacobian = (lensVertexCopyShift ? 1.0f : (shiftedNeighbor.lensVertexJacobian / neighborReconnection.lensVertexJacobian)) *
                                                          (shiftedNeighbor.secondaryPathJacobian / neighborReconnection.secondaryPathJacobian);

                float neighborWeight = luminance(shiftedNeighbor.radiance) * shiftedJacobian * (useConfidenceWeights ? centralReservoir.confidence : 1.0f);
                neighborPHat = isnan(neighborWeight) ? float3(0.0f, 0.0f, 0.0f) : shiftedNeighbor.radiance;
                shiftedJacobian = isnan(neighborWeight) ? 1.0f : shiftedJacobian; // W_{y_i} = W_{x_i} * |T(x)|
                neighborWeight = isnan(neighborWeight) ? 0.0f : neighborWeight / neighborCount;

                // Update the relevant data.
                neighborReconnection.update(shiftedNeighbor);

                float m2 = luminance(neighborReservoir.integrand) * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);

                neighborSampleMIS = (m2 + neighborWeight > 0.0f) ? m2 / (neighborWeight + m2) : 0.0f;
            }

            bool neighborSelected = dstReservoir.addSampleFromReservoir(sg, neighborSampleMIS, neighborPHat, shiftedJacobian, neighborReservoir);
            dstReconnectionData = neighborSelected ? neighborReconnection : dstReconnectionData;
        }

        // Consider the central sample at the end (which required a sum over the neighbor samples).
        bool centralSelected = dstReservoir.addSampleFromReservoir(sg, centralSampleMIS, centralReservoir.integrand, 1.0f, centralReservoir);
        dstReconnectionData = centralSelected ? centralReconnection : dstReconnectionData;

        // After everything is done, account for the final MIS weighting of 1 / M.
        dstReservoir.totalWeight /= (validNeighbors + 1);

        currReservoirs[reservoirIdx] = dstReservoir;
        currReconnectionData[reservoirIdx] = dstReconnectionData;
    }
}

cbuffer CB
{
    SpatialResampling gSpatialResampling;
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
    gSpatialResampling.run(pixel);
}
