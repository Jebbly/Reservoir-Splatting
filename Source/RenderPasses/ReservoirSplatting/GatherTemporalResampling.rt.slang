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

import GatherHelper;

import ShiftOptions;
import ShiftMapping;

struct GatherTemporalResampling
{
    bool useConfidenceWeights;

    PathTracerParams params;

    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PathReservoir> currReservoirs;

    StructuredBuffer<ReconnectionData> prevReconnectionData;
    RWStructuredBuffer<ReconnectionData> currReconnectionData;

    static const float2 kNeighborOffsets[4] = {
        float2(0, 0), float2(1, 0),
        float2(0, 1), float2(1, 1)
    };

    Texture2D<PackedHitInfo> vbuffer;

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        print("======= Gather Temporal Resampling =======", pixel);
        SampleGenerator sg = SampleGenerator(pixel, (params.useFixedSeed > 0) ? params.fixedSeed + 3 : kNumRenderPasses * (params.frameCount + params.seed) + 3);

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

        const bool doLensVertexCopy = (depthOfFieldProbs.x > 0.0f);
        const bool doPrimaryHitReconnection = (depthOfFieldProbs.y > 0.0f);

        // At this point we can assume that a previous reservoir exists and is valid.
        PathReservoir dstReservoir = PathReservoir();
        ReconnectionData dstReconnectionData = ReconnectionData();

        GatherHelper gatherHelper = GatherHelper(pixel, params.frameDim);
        if (doLensVertexCopy)
        {
            PathReservoir prevReservoir = prevReservoirs[reservoirIdx];
            PathReservoir currReservoir = currReservoirs[reservoirIdx];

            ReconnectionData prevReconnection = prevReconnectionData[reservoirIdx];
            ReconnectionData currReconnection = currReconnectionData[reservoirIdx];

            float currSampleMIS = 0.0f;
            float3 currPHat = float3(0.0f, 0.0f, 0.0f);
            // Handle the current reservoir (initial candidate) first.
            if (any(currReservoir.integrand > 0.0f))
            {
                // Shifting current sample to current domain... pHat remains the same and the Jacobian is 1.
                float m1 = luminance(currReservoir.integrand) * (useConfidenceWeights ? currReservoir.confidence : 1.0f);
                currPHat = currReservoir.integrand;

                // Shifting current sample to previous domain.
                float2 relativeSubPixel = currReservoir.subPixel;
                float2 floatingCoord = gatherHelper.getFloatingCoords();
                ShiftedPathData shiftedCurr = gatherLensVertexCopyShift(sg, currReconnection, currReconnection.time + gCameraManager.artificialFrameTime, floatingCoord + relativeSubPixel, currReconnection.lensSample);
                float shiftedJacobian = shiftedCurr.secondaryPathJacobian / currReconnection.secondaryPathJacobian;
                float m2 = luminance(shiftedCurr.radiance) * shiftedJacobian * (useConfidenceWeights ? prevReservoir.confidence : 1.0f) * depthOfFieldProbs.x;

                // TODO: This check is only necessary if the secondaryPathJacobian ever equals 0... is that ever possible?
                m2 = isnan(m2) ? 0.0f : m2;

                currSampleMIS = (m1 + m2 == 0.0f) ? 0.0f : m1 / (m1 + m2);
            }

            bool currSelected = dstReservoir.addSampleFromReservoir(sg, currSampleMIS, currPHat, 1.0f, currReservoir);
            dstReconnectionData = currSelected ? currReconnection : dstReconnectionData;

            // Reconnection shift. Connect y_1 with x_2, regardless of if it's in the shutter time or not.
            float prevSampleMIS = 0.0f;
            float3 prevPHat = float3(0.0f, 0.0f, 0.0f);
            float shiftedJacobian = 1.0f;
            if (any(prevReservoir.integrand > 0.0f))
            {
                // Shift previous sample to current domain.
                ShiftedPathData shiftedPrev = gatherLensVertexCopyShift(sg, prevReconnection, prevReconnection.time, float2(pixel) + prevReservoir.subPixel, prevReconnection.lensSample);
                shiftedJacobian = shiftedPrev.secondaryPathJacobian / prevReconnection.secondaryPathJacobian;
                float m1 = luminance(shiftedPrev.radiance) * shiftedJacobian * (useConfidenceWeights ? currReservoir.confidence : 1.0f);

                // TODO: are all these NaN-checks still necessary? Should only happen if the secondary path Jacobian is 0 at any point.
                prevPHat = isnan(m1) ? float3(0.0f, 0.0f, 0.0f) : shiftedPrev.radiance;
                shiftedJacobian = isnan(m1) ? 1.0f : shiftedJacobian; // W_{y_i} = W_{x_i} * |T(x)|
                m1 = isnan(m1) ? 0.0f : m1;

                // Update the relevant data.
                prevReconnection.update(shiftedPrev);

                // Shift previous sample to previous domain... the previous pHat is the same with a Jacobian of 1.
                float m2 = luminance(prevReservoir.integrand) * (useConfidenceWeights ? prevReservoir.confidence : 1.0f) * depthOfFieldProbs.x;

                prevSampleMIS = ((m1 + m2) == 0.0f) ? 0.0f : m2 / (m1 + m2);
            }

            bool prevSelected = dstReservoir.addSampleFromReservoir(sg, prevSampleMIS, prevPHat, shiftedJacobian, prevReservoir);
            dstReconnectionData = prevSelected ? prevReconnection : dstReconnectionData;
        }

        // If depth of field is enabled, we perform an additional GRIS pass.
        if (hasDepthOfField && doPrimaryHitReconnection)
        {
            // If we did the lens vertex copy, we should continue from the destination reservoir.
            // Otherwise, start from scratch and load the stored current reservoir.
            PathReservoir prevReservoir = prevReservoirs[reservoirIdx];
            PathReservoir currReservoir = doLensVertexCopy ? dstReservoir : currReservoirs[reservoirIdx];

            ReconnectionData prevReconnection = prevReconnectionData[reservoirIdx];
            ReconnectionData currReconnection = doLensVertexCopy ? dstReconnectionData : currReconnectionData[reservoirIdx];

            dstReservoir = PathReservoir();
            dstReconnectionData = ReconnectionData();

            float currSampleMIS = 0.0f;
            float3 currPHat = float3(0.0f, 0.0f, 0.0f);
            // Handle the current reservoir (initial candidate) first.
            if (any(currReservoir.integrand > 0.0f))
            {
                // Shifting current sample to current domain... pHat remains the same and the Jacobian is 1.
                float m1 = luminance(currReservoir.integrand) * (useConfidenceWeights ? currReservoir.confidence : 1.0f);
                currPHat = currReservoir.integrand;

                // Shifting current sample to previous domain.
                float2 relativeSubPixel = currReservoir.subPixel;
                float2 floatingCoord = gatherHelper.getFloatingCoords();
                ShiftedPathData shiftedCurr = gatherPrimaryHitReconnectionShift(sg, currReconnection, currReconnection.time + gCameraManager.artificialFrameTime, floatingCoord + relativeSubPixel, currReconnection.firstHit);
                float shiftedJacobian = (shiftedCurr.lensVertexJacobian * shiftedCurr.secondaryPathJacobian) / (currReconnection.lensVertexJacobian * currReconnection.secondaryPathJacobian);
                float m2 = luminance(shiftedCurr.radiance) * shiftedJacobian * (useConfidenceWeights ? prevReservoir.confidence : 1.0f) * depthOfFieldProbs.y;

                // TODO: This check is only necessary if the secondaryPathJacobian ever equals 0... is that ever possible?
                m2 = isnan(m2) ? 0.0f : m2;

                currSampleMIS = (m1 + m2 == 0.0f) ? 0.0f : m1 / (m1 + m2);
            }

            bool currSelected = dstReservoir.addSampleFromReservoir(sg, currSampleMIS, currPHat, 1.0f, currReservoir);
            dstReconnectionData = currSelected ? currReconnection : dstReconnectionData;

            // Reconnection shift. Connect y_1 with x_2, regardless of if it's in the shutter time or not.
            float prevSampleMIS = 0.0f;
            float3 prevPHat = float3(0.0f, 0.0f, 0.0f);
            float shiftedJacobian = 1.0f;
            if (any(prevReservoir.integrand > 0.0f))
            {
                // Shift previous sample to current domain.
                ShiftedPathData shiftedPrev = gatherPrimaryHitReconnectionShift(sg, prevReconnection, prevReconnection.time, float2(pixel) + prevReservoir.subPixel, prevReconnection.firstHit);
                shiftedJacobian = (shiftedPrev.lensVertexJacobian * shiftedPrev.secondaryPathJacobian) / (prevReconnection.lensVertexJacobian * prevReconnection.secondaryPathJacobian);
                float m1 = luminance(shiftedPrev.radiance) * shiftedJacobian * (useConfidenceWeights ? currReservoir.confidence : 1.0f);

                // TODO: are all these NaN-checks still necessary? Should only happen if the secondary path Jacobian is 0 at any point.
                prevPHat = isnan(m1) ? float3(0.0f, 0.0f, 0.0f) : shiftedPrev.radiance;
                shiftedJacobian = isnan(m1) ? 1.0f : shiftedJacobian; // W_{y_i} = W_{x_i} * |T(x)|
                m1 = isnan(m1) ? 0.0f : m1;

                // Update the relevant data.
                prevReconnection.update(shiftedPrev);

                // Shift previous sample to previous domain... the previous pHat is the same with a Jacobian of 1.
                float m2 = luminance(prevReservoir.integrand) * (useConfidenceWeights ? prevReservoir.confidence : 1.0f) * depthOfFieldProbs.y;

                prevSampleMIS = ((m1 + m2) == 0.0f) ? 0.0f : m2 / (m1 + m2);
            }

            bool prevSelected = dstReservoir.addSampleFromReservoir(sg, prevSampleMIS, prevPHat, shiftedJacobian, prevReservoir);
            dstReconnectionData = prevSelected ? prevReconnection : dstReconnectionData;
        }

        currReservoirs[reservoirIdx] = dstReservoir;
        currReconnectionData[reservoirIdx] = dstReconnectionData;
    }
}

cbuffer CB
{
    GatherTemporalResampling gGatherTemporalResampling;
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
    gGatherTemporalResampling.run(pixel);
}
