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

struct ScatterBackupTemporalResampling
{
    bool useConfidenceWeights;

    PathTracerParams params;

    ScatterBackupMIS misOption;

    StructuredBuffer<uint32_t> cellCounters;    // Number of reservoirs that were scattered into a given cell.
    StructuredBuffer<uint> cellOffsets;         // Where the initial cell offset begins.
    StructuredBuffer<uint2> sortedReservoirs;   // Linearized buffer of scattered reservoirs.

    StructuredBuffer<PathReservoir> backupReservoirs;
    StructuredBuffer<PathReservoir> prevReservoirs;
    RWStructuredBuffer<PathReservoir> currReservoirs;

    StructuredBuffer<ReconnectionData> backupReconnectionData;
    StructuredBuffer<ReconnectionData> prevReconnectionData;
    RWStructuredBuffer<ReconnectionData> currReconnectionData;

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        print("======= Scatter + Backup Temporal Resampling =======", pixel);

        // TODO: use a better seed here.
        SampleGenerator sg = SampleGenerator(pixel, (params.useFixedSeed > 0) ? params.fixedSeed + 6 : kNumRenderPasses * (params.frameCount + params.seed) + 6);

        PathReservoir dstReservoir = PathReservoir();
        ReconnectionData dstReconnectionData = ReconnectionData();

        uint32_t reservoirIdx = params.linearizePixel(pixel);

        PathReservoir currReservoir = currReservoirs[reservoirIdx];
        ReconnectionData currReconnection = currReconnectionData[reservoirIdx];

        PathReservoir backupReservoir = backupReservoirs[reservoirIdx];
        ReconnectionData backupReconnection = backupReconnectionData[reservoirIdx];

        float misScale = (misOption == ScatterBackupMIS::Pairwise) ? 0.5f : 1.0f;

        GatherHelper gatherHelper = GatherHelper(pixel, params.frameDim);

        // We first consider the current reservoir (the initial sample).
        {
            float currSampleMIS = 1.0f;
            if (any(currReservoir.integrand > 0.0f))
            {
                float m1 = luminance(currReservoir.integrand) * (useConfidenceWeights ? currReservoir.confidence : 1.0f);

                // Shift the current sample to the previous frame domain by reprojection.
                float m2 = 0.0f;
                ShiftedPathData scatteredCurr = scatterReprojectionShift(sg, currReconnection, currReconnection.time + gCameraManager.artificialFrameTime, currReconnection.firstHit, currReconnection.lensSample);
                float scatteredJacobian = (scatteredCurr.subPixelJacobian * scatteredCurr.secondaryPathJacobian) / (currReconnection.subPixelJacobian * currReconnection.secondaryPathJacobian);
                m2 = luminance(scatteredCurr.radiance) * scatteredJacobian;
                m2 = isnan(m2) ? 0.0f : m2;

                int2 scatteredPixel = int2(floor(scatteredCurr.fractionalPixel));
                if (params.validateIntegerPixelBounds(scatteredPixel))
                {
                    PathReservoir prevReservoir = prevReservoirs[params.linearizePixel(scatteredPixel)];
                    m2 *= 0.5f * (useConfidenceWeights ? prevReservoir.confidence : 1.0f);
                }
                else
                {
                    m2 = 0.0f;
                }

                // Shift the current sample to the previous frame domain.
                float m3 = 0.0f;
                float2 relativeSubPixel = currReservoir.subPixel;
                float2 floatingCoord = gatherHelper.getFloatingCoords();

                if (params.validateFloatingPixelBounds(floatingCoord))
                {
                    ShiftedPathData shiftedCurr = gatherLensVertexCopyShift(sg, currReconnection, currReconnection.time + gCameraManager.artificialFrameTime, floatingCoord + relativeSubPixel, currReconnection.lensSample);
                    float shiftedJacobian = (shiftedCurr.secondaryPathJacobian / currReconnection.secondaryPathJacobian);
                    m3 = luminance(shiftedCurr.radiance) * shiftedJacobian * 0.5f * (useConfidenceWeights ? backupReservoir.confidence : 1.0f);
                }

                // TODO: This check is only necessary if the secondaryPathJacobian ever equals 0... is that ever possible?
                m3 = isnan(m3) ? 0.0f : m3;

                switch (misOption)
                {
                    case ScatterBackupMIS::Balance:
                    {
                        currSampleMIS = (m1 + m2 + m3 == 0.0f) ? 0.0f : m1 / (m1 + m2 + m3);
                        break;
                    }
                    case ScatterBackupMIS::Pairwise:
                    {
                        currSampleMIS = 0.5f * (((m1 + m2 == 0.0f) ? 0.0f : 0.5f * m1 / (0.5f * m1 + m2)) +
                                                ((m1 + m3 == 0.0f) ? 0.0f : 0.5f * m1 / (0.5f * m1 + m3)));
                        break;
                    }
                }
            }

            bool currSelected = dstReservoir.addSampleFromReservoir(sg, currSampleMIS, currReservoir.integrand, 1.0f, currReservoir);
            dstReconnectionData = currSelected ? currReconnection : dstReconnectionData;
        }

        // Next, consider the backup sample.
        {
            float backupSampleMIS = 0.0f;
            float3 backupPHat = float3(0.0f, 0.0f, 0.0f);
            float shiftedJacobian = 1.0f;
            if (any(backupReservoir.integrand > 0.0f))
            {
                // Shift previous sample to current domain.
                ShiftedPathData shiftedBackup = gatherLensVertexCopyShift(sg, backupReconnection, backupReconnection.time, float2(pixel) + backupReservoir.subPixel, backupReconnection.lensSample);
                shiftedJacobian = (shiftedBackup.secondaryPathJacobian / backupReconnection.secondaryPathJacobian);
                float m1 = luminance(shiftedBackup.radiance) * shiftedJacobian * (useConfidenceWeights ? currReservoir.confidence : 1.0f);

                // TODO: are all these NaN-checks still necessary? Should only happen if the secondary path Jacobian is 0 at any point.
                backupPHat = isnan(m1) ? float3(0.0f, 0.0f, 0.0f) : shiftedBackup.radiance;
                shiftedJacobian = isnan(m1) ? 1.0f : shiftedJacobian; // W_{y_i} = W_{x_i} * |T(x)|
                m1 = isnan(m1) ? 0.0f : m1;

                // Update the relevant data.
                backupReconnection.update(shiftedBackup);

                float m2 = 0.0f;
                if (misOption == ScatterBackupMIS::Balance)
                {
                    ShiftedPathData scatteredBackup = scatterReprojectionShift(sg, backupReconnection, backupReconnection.time + gCameraManager.artificialFrameTime, backupReconnection.firstHit, backupReconnection.lensSample);
                    float scatteredJacobian = (scatteredBackup.subPixelJacobian * scatteredBackup.secondaryPathJacobian) / (backupReconnection.subPixelJacobian * backupReconnection.secondaryPathJacobian);
                    m2 = luminance(scatteredBackup.radiance) * scatteredJacobian * shiftedJacobian;
                    m2 = isnan(m2) ? 0.0f : m2;

                    int2 scatteredPixel = int2(floor(scatteredBackup.fractionalPixel));
                    if (params.validateIntegerPixelBounds(scatteredPixel))
                    {
                        PathReservoir prevReservoir = prevReservoirs[params.linearizePixel(scatteredPixel)];
                        m2 *= 0.5f * (useConfidenceWeights ? prevReservoir.confidence : 1.0f);
                    }
                    else
                    {
                        m2 = 0.0f;
                    }
                }

                // Shift previous sample to previous domain... the previous pHat is the same with a Jacobian of 1.
                float m3 = luminance(backupReservoir.integrand) * 0.5f * (useConfidenceWeights ? backupReservoir.confidence : 1.0f);

                backupSampleMIS = misScale * ((m1 + m2 + m3 == 0.0f) ? 0.0f : m3 / (misScale * m1 + m2 + m3));
            }

            bool backupSelected = dstReservoir.addSampleFromReservoir(sg, backupSampleMIS, backupPHat, shiftedJacobian, backupReservoir);
            dstReconnectionData = backupSelected ? backupReconnection : dstReconnectionData;
        }

        // Track the confidence after the backup sample.
        float newConfidence = dstReservoir.confidence;

        // Lastly, consider the scattered reservoirs from the previous frame.
        uint32_t numReservoirs = cellCounters[reservoirIdx];
        uint32_t cellOffset = cellOffsets[reservoirIdx];
        for (uint32_t i = 0; i < numReservoirs; i++)
        {
            uint2 scatteredPixel = sortedReservoirs[cellOffset + i];
            // Anything that was scattered should already be valid.
            uint32_t scatteredIdx = params.linearizePixel(scatteredPixel);
            PathReservoir prevReservoir = prevReservoirs[scatteredIdx];
            ReconnectionData prevReconnection = prevReconnectionData[scatteredIdx];

            float prevSampleMIS = 0.0f;
            float3 prevPHat = float3(0.0f, 0.0f, 0.0f);
            float scatteredJacobian = 1.0f;
            if (any(prevReservoir.integrand > 0.0f))
            {
                // Scatter into the current domain.
                ShiftedPathData scatteredPrev = scatterReprojectionShift(sg, prevReconnection, prevReconnection.time, prevReconnection.firstHit, prevReconnection.lensSample, true);
                scatteredJacobian = (scatteredPrev.subPixelJacobian * scatteredPrev.secondaryPathJacobian) / (prevReconnection.subPixelJacobian * prevReconnection.secondaryPathJacobian);
                float m1 = luminance(scatteredPrev.radiance) * scatteredJacobian * (useConfidenceWeights ? currReservoir.confidence : 1.0f);
                m1 = isnan(m1) ? 0.0f : m1;
                prevPHat = isnan(m1) ? float3(0.0f, 0.0f, 0.0f) : scatteredPrev.radiance;
                scatteredJacobian = isnan(m1) ? 1.0f : scatteredJacobian; // W_{y_i} = W_{x_i} * |T(x)|

                // Update the relevant data.
                prevReconnection.update(scatteredPrev);

                // The scattered reservoir will also be at a new subpixel location, compared to where it originally started.
                prevReservoir.setSubPixel(prevReconnection.subPixel);

                // m2 is based on the scattered reservoir from the previous domain.
                float m2 = luminance(prevReservoir.integrand) * 0.5f * (useConfidenceWeights ? prevReservoir.confidence : 1.0f);

                // Shift with the domain duplication.
                float m3 = 0.0f;
                if (misOption == ScatterBackupMIS::Balance)
                {
                    float2 relativeSubPixel = prevReservoir.subPixel;
                    float2 floatingCoord = gatherHelper.getFloatingCoords();

                    if (params.validateFloatingPixelBounds(floatingCoord + relativeSubPixel))
                    {
                        ShiftedPathData shiftedPrev = gatherLensVertexCopyShift(sg, prevReconnection, prevReconnection.time + gCameraManager.artificialFrameTime, floatingCoord + relativeSubPixel, prevReconnection.lensSample);
                        float shiftedJacobian = (shiftedPrev.secondaryPathJacobian / prevReconnection.secondaryPathJacobian);
                        m3 = luminance(shiftedPrev.radiance) * shiftedJacobian * scatteredJacobian * 0.5f * (useConfidenceWeights ? backupReservoir.confidence : 1.0f);
                    }
                    
                    // TODO: This check is only necessary if the secondaryPathJacobian ever equals 0... is that ever possible?
                    m3 = isnan(m3) ? 0.0f : m3;
                }

                prevSampleMIS = misScale * ((m1 + m2 + m3 == 0.0f) ? 0.0f : m2 / (misScale * m1 + m2 + m3));
            }

            bool prevSelected = dstReservoir.addSampleFromReservoir(sg, prevSampleMIS, prevPHat, scatteredJacobian, prevReservoir);
            dstReconnectionData = prevSelected ? prevReconnection : dstReconnectionData;
        }

        // We use the motion vector to compute the new confidence weight (same as Area ReSTIR).
        int2 prevPixelTopLeft = int2(floor(gatherHelper.prevPixel));
        float2 fractionalCoord = saturate(gatherHelper.prevPixel - float2(prevPixelTopLeft));
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
    ScatterBackupTemporalResampling gScatterBackupTemporalResampling;
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
    gScatterBackupTemporalResampling.run(pixel);
}
