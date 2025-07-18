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
import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import LoadShadingData;

import CameraManager;
import ReconnectionData;
import Reservoir;

import Params;
import ShiftOptions;

struct CollectTemporalSamples
{
    PathTracerParams params;
    bool useConfidenceWeights;

    Texture2D<float2> motionVectors;
    GatherMechanism gatherOption;

    StructuredBuffer<ShiftedPathData> shiftedPaths; // Only valid during full robust reuse.
    RWTexture2D<float2> floatingCoords;

    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PathReservoir> intermediateReservoirs;

    StructuredBuffer<ReconnectionData> prevReconnectionData;
    RWStructuredBuffer<ReconnectionData> intermediateReconnectionData;

    static const uint2 kPrevNeighborOffsets[4] = {
        uint2(0, 0), uint2(1, 0),
        uint2(0, 1), uint2(1, 1)
    };

    /**
     * Creates a "previous reservoir" corresponding to the shift mapping and reuse option,
     * basically identifying what "final" sample to resample during temporal resampling.
     * @param[in] currPixel Current pixel coordinate.
     */
    void execute(const uint2 currPixel)
    {
        if (!params.validateIntegerPixelBounds(currPixel)) return;
        printSetPixel(currPixel);
        print("======= Collect Temporal Samples =======", currPixel);

        // TODO: use a better seed here.
        SampleGenerator sg = SampleGenerator(currPixel, (params.useFixedSeed > 0) ? params.fixedSeed + 2 : kNumRenderPasses * (params.frameCount + params.seed) + 2);
        uint32_t currReservoirIndex = params.linearizePixel(currPixel);

        // Other shifts use the motion vector to check where the current pixel was in the previous frame.
        float2 motionVector = motionVectors[currPixel];
        // TODO: without this band-aid, the hybrid + fast will diverge with static camera / scene.
        motionVector = (length(motionVector) < 1e-06) ? float2(0.0f, 0.0f) : motionVector; 
        float2 prevPixel = float2(currPixel) + motionVector * float2(params.frameDim);

        // print("Current Pixel", currPixel);
        // print("Previous Pixel", prevPixel);

        PathReservoir dstReservoir = PathReservoir();
        ReconnectionData dstReconnectionData = ReconnectionData();
        float2 dstFloatingCoord = float2(-1.0f, -1.0f);

        if (!params.validateFloatingPixelBounds(prevPixel))
        {
            intermediateReservoirs[currReservoirIndex] = dstReservoir;
            intermediateReconnectionData[currReservoirIndex] = dstReconnectionData;
            floatingCoords[currPixel] = dstFloatingCoord;
            return;
        }

        // If the pixel falls within the image bounds, we should be able to safely cast it to a uint2.
        int2 prevPixelTopLeft = int2(floor(prevPixel));
        float2 fractionalCoord = saturate(prevPixel - float2(prevPixelTopLeft));
        // print("Fractional Coord", fractionalCoord);

        // In both the fast and robust case, the current pixel falls into a 2x2 area of the previous image.
        // We thus need to iterate over all possibilities to select a single sample to shift between.
        switch (gatherOption)
        {
            case (GatherMechanism::Fast):
            {
                // Fast option uses the floating reservoir corresponding exactly to the fractional motion vector.
                dstFloatingCoord = prevPixel;

                // Resample between all 4 possibilities.
                float totalConfidence = 0.0f;
                for (int x = 0; x < 2; x++)
                {
                    for (int y = 0; y < 2; y++)
                    {
                        float2 offset = float2(x, y);
                        uint2 neighborPixel = prevPixelTopLeft + uint2(offset);
                        if (!params.validateIntegerPixelBounds(neighborPixel)) continue;

                        uint32_t neighborIndex = params.linearizePixel(neighborPixel);
                        PathReservoir neighborReservoir = prevReservoirs[neighborIndex];
                        float2 relativeSubPixel = offset + neighborReservoir.subPixel - fractionalCoord;

                        float bilinearWeight = lerp(1 - x, x, fractionalCoord.x) * lerp(1 - y, y, fractionalCoord.y);
                        totalConfidence += bilinearWeight * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);

                        if (all(relativeSubPixel >= 0.0f && relativeSubPixel < 1.0f))
                        {
                            neighborReservoir.setSubPixel(relativeSubPixel);
                            bool selected = dstReservoir.addSampleFromReservoir(sg, 1.0f, neighborReservoir.integrand, 1.0f, neighborReservoir);
                            dstReconnectionData = selected ? prevReconnectionData[neighborIndex] : dstReconnectionData;
                        }
                    }
                }
                dstReservoir.confidence = totalConfidence;
                break;
            }
            case (GatherMechanism::Clamped):
            {
                // Fast robust option will clamp the "floating" reservoir to the nearest integer pixel.
                uint2 roundedPrevPixel = uint2(round(prevPixel));
                dstFloatingCoord = float2(roundedPrevPixel);

                if (!params.validateIntegerPixelBounds(roundedPrevPixel)) break;

                uint32_t prevReservoirIdx = params.linearizePixel(roundedPrevPixel);
                dstReservoir = prevReservoirs[prevReservoirIdx];
                dstReconnectionData = prevReconnectionData[prevReservoirIdx];

                break;
            }
            case (GatherMechanism::Robust):
            {
                // Full robust option also uses the floating reservoir corresponding exactly to the fractional motion vector.
                dstFloatingCoord = prevPixel;

                // Resample between all 4 possibilities. In this case, all neighbor pixels have some way to fit in the floating reservoir,
                // but we might have to examine the cases where they have to be shifted in order to fit.
                float totalConfidence = 0.0f;
                for (int x = 0; x < 2; x++)
                {
                    for (int y = 0; y < 2; y++)
                    {
                        float2 offset = float2(x, y);
                        uint2 neighborPixel = prevPixelTopLeft + uint2(offset);
                        if (!params.validateIntegerPixelBounds(neighborPixel)) continue;

                        uint32_t neighborIndex = params.linearizePixel(neighborPixel);
                        PathReservoir neighborReservoir = prevReservoirs[neighborIndex];
                        ReconnectionData neighborReconnection = prevReconnectionData[neighborIndex];
                        float2 relativeSubPixel = offset + neighborReservoir.subPixel - fractionalCoord;
                        float bilinearWeight = lerp(1 - x, x, fractionalCoord.x) * lerp(1 - y, y, fractionalCoord.y);
                        totalConfidence += bilinearWeight * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);

                        // Compute the appropriate offset to fit the sample into the floating reservoir.
                        int2 requiredOffset = int2(0, 0);
                        if (relativeSubPixel.x < 0.0f) requiredOffset.x += 1;
                        if (relativeSubPixel.x >= 1.0f) requiredOffset.x -= 1;
                        if (relativeSubPixel.y < 0.0f) requiredOffset.y += 1;
                        if (relativeSubPixel.y >= 1.0f) requiredOffset.y -= 1;
                        relativeSubPixel += requiredOffset;

                        requiredOffset += int2(1, 1);
                        uint offsetIndex = requiredOffset.x + 3 * requiredOffset.y;
                        // (0, 0) means that no shift was required.
                        bool noShiftNeeded = (offsetIndex == 4);
                        offsetIndex = (offsetIndex > 4) ? offsetIndex - 1 : offsetIndex;
                        ShiftedPathData shiftedPath = noShiftNeeded ? 
                            ShiftedPathData(neighborReconnection.firstHit, float2(neighborPixel) + neighborReservoir.subPixel, neighborReconnection.lensSample, -neighborReconnection.firstWi,
                                            neighborReconnection.subPixelJacobian, neighborReconnection.lensVertexJacobian, neighborReconnection.secondaryPathJacobian, neighborReservoir.integrand) 
                            : shiftedPaths[8 * neighborIndex + offsetIndex];
                        float shiftedJacobian = noShiftNeeded ? 1.0f : shiftedPath.secondaryPathJacobian / neighborReconnection.secondaryPathJacobian;

                        // Compute the MIS weight of the current sample (generalized balance heuristic w/ bilinear weighting).
                        float pHatSource = luminance(neighborReservoir.integrand);
                        float sourceWeight = bilinearWeight * pHatSource * (useConfidenceWeights ? neighborReservoir.confidence : 1.0f);
                        float totalPHat = sourceWeight;
                        for (int _x = 0; _x < 2; ++_x)
                        {
                            for (int _y = 0; _y < 2; ++_y)
                            {
                                int2 tempOffset = int2(_x, _y);
                                uint2 tempOffsetPixel = prevPixelTopLeft + tempOffset;
                                if (!params.validateIntegerPixelBounds(tempOffsetPixel)) continue;
                                int2 diff = tempOffset - int2(offset);
                                if (all(diff == 0)) continue;

                                float tempBilinearWeight = lerp(1 - _x, _x, fractionalCoord.x) * lerp(1 - _y, _y, fractionalCoord.y);

                                int2 temp = diff + int2(1, 1);
                                uint tempIndex = temp.x + 3 * temp.y;
                                tempIndex = (tempIndex > 4) ? tempIndex - 1 : tempIndex;
                                ShiftedPathData tempPath = shiftedPaths[8 * neighborIndex + tempIndex];

                                float tempPHat = luminance(tempPath.radiance);
                                float tempJacobian = tempPath.secondaryPathJacobian / neighborReconnection.secondaryPathJacobian;
                                float tempConfidence = useConfidenceWeights ? prevReservoirs[params.linearizePixel(tempOffsetPixel)].confidence : 1.0f;

                                totalPHat += tempBilinearWeight * tempPHat * tempJacobian * tempConfidence;
                            }
                        }
                        float misWeight = (totalPHat > 0.0f) ? sourceWeight / totalPHat : 0.0f;

                        // Update the reservoir / reconnection to be relevant in the current domain.
                        neighborReservoir.setSubPixel(relativeSubPixel);
                        neighborReconnection.update(shiftedPath);
                        neighborReconnection.subPixel = relativeSubPixel; // This is handled specially because it is a fractional reservoir.

                        bool selected = dstReservoir.addSampleFromReservoir(sg, misWeight, luminance(shiftedPath.radiance), shiftedJacobian, neighborReservoir);
                        dstReconnectionData = selected ? neighborReconnection : dstReconnectionData;
                    }
                }
                dstReservoir.confidence = totalConfidence;
                break;
            }
        }

        // TODO: this is a final catch, but we should really be more robust than this.
        if (isnan(dstReservoir.totalWeight))
        {
            dstReservoir.integrand = float3(0.0f, 0.0f, 0.0f);
            dstReservoir.totalWeight = 0.0f;
            dstReconnectionData.subPixelJacobian = 1.0f;
            dstReconnectionData.lensVertexJacobian = 1.0f;
            dstReconnectionData.secondaryPathJacobian = 1.0f;
            dstReconnectionData.pathLength = 0;
        }

        intermediateReservoirs[currReservoirIndex] = dstReservoir;
        intermediateReconnectionData[currReservoirIndex] = dstReconnectionData;
        floatingCoords[currPixel] = dstFloatingCoord;
    }
};

cbuffer CB
{
    CollectTemporalSamples gCollectTemporalSamples;
};

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    gCollectTemporalSamples.execute(dispatchThreadId.xy);
}
