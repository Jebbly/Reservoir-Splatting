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
 
#include "Scene/SceneDefines.slangh"
#include "RaytracingCommon.slang"

import Rendering.Utils.PixelStats;
import Utils.Debug.PixelDebug;
import Utils.Math.PackedFormats;
import Utils.Math.FormatConversion;
import PathTracer;
import ReconnectionData;
import Reservoir;

import Utils.Color.ColorHelpers;

import Utils.NVAPI; // TODO(@skallweit) this is currently needed to make g_NvidiaExt visible

ParameterBlock<PathTracer> gPathTracer;

struct InitialCandidates
{
    PathTracerParams params;
    RWStructuredBuffer<PathReservoir> currReservoirs;
    RWStructuredBuffer<ReconnectionData> currReconnectionData;

    void addCandidateReservoir(inout PathState path)
    {
        const bool isFirstSample = (path.getSampleIdx() == (kSamplesPerPixel - 1));
        const uint32_t reservoirIdx = params.linearizePixel(path.getPixel());

        // If this is the first sample, we initialize an empty PathReservoir.
        PathReservoir existingReservoir = isFirstSample ? PathReservoir() : currReservoirs[reservoirIdx]; 
        bool selected = existingReservoir.add(path.sg, 1.0f / float(kSamplesPerPixel), path.reservoir); // All samples are IID, so m_i = 1 / M.
        existingReservoir.subPixel = selected ? path.reconnection.subPixel : existingReservoir.subPixel;
        existingReservoir.confidence = 1.0f;
        currReservoirs[reservoirIdx] = existingReservoir;

        // If this is the first sample, also use a placeholder ReconnectionData.
        ReconnectionData existingReconnection = isFirstSample ? ReconnectionData() : currReconnectionData[reservoirIdx]; 
        currReconnectionData[reservoirIdx] = selected ? path.reservoir.selectedReconnection : existingReconnection;

        print("Path Reservoir Selected", selected);
        print("Existing Reservoir Integrand", existingReservoir.integrand);
        print("Existing Reservoir PHat", luminance(existingReservoir.integrand));
        print("Existing Reservoir Total Weight", existingReservoir.totalWeight);
        print("Existing Reservoir UCW", existingReservoir.computeUCW());
        // print("Existing Reservoir Color", existingReservoir.integrand * existingReservoir.computeUCW());
        // print("Size of Path", sizeof(PathState));
    }

    /** Traces a path and write result to output.
        \param[in] pathID Path ID.
    */
    void tracePath(uint pathID)
    {
        PathState path = {};

        gPathTracer.generatePath(path, (params.useFixedSeed > 0) ? params.fixedSeed + 0 : kNumRenderPasses * (params.frameCount + params.seed) + 0, pathID);
        gPathTracer.setupPathLogging(path);

        // Handle primary hit/miss.
        gPathTracer.nextHit(path);
        if (path.isHit())
        {
            gPathTracer.handleHit(path);
        }
        else
        {
            // Note that there is no GeneratePaths pass in this case, so missed primary hits need to be handled here.
            gPathTracer.handlePrimaryMiss(path);
            addCandidateReservoir(path);
            return;
        }

        // Trace path.
        while (path.isActive())
        {
            gPathTracer.nextHit(path);
            if (path.isHit())
            {
                gPathTracer.handleHit(path);
            }
            else
            {
                gPathTracer.handleMiss(path);
            }
        }

        addCandidateReservoir(path);
    }

    /** Runs the path tracer for a single pixel.
        \param[in] pixel Pixel index.
    */
    void run(uint2 pixel)
    {
        printSetPixel(pixel);
        print("======= Initial Candidate Generation =======", pixel);

        if (kSamplesPerPixel == 1)
        {
            // Handle fixed 1 spp case.
            uint pathID = pixel.x | (pixel.y << 12);
            tracePath(pathID);
        }
        else // (kSamplesPerPixel > 1)
        {
            // Handle fixed multiple spp case.
            for (uint sampleIdx = 0; sampleIdx < kSamplesPerPixel; ++sampleIdx)
            {
                uint pathID = pixel.x | (pixel.y << 12) | (sampleIdx << 24);
                tracePath(pathID);
            }
        }
    }
}

cbuffer CB
{
    InitialCandidates gInitialCandidates;
}

[numthreads(8, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (all(pixel >= gPathTracer.params.frameDim)) return;

    gInitialCandidates.run(pixel);
}
