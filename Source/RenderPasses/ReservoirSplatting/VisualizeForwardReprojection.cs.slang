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

import Utils.Debug.PixelDebug;

import Params;

struct VisualizeForwardReprojection
{
    PathTracerParams params;
    bool isMultiScatter;
    StructuredBuffer<uint32_t> cellCounters;
    StructuredBuffer<uint32_t> multiCellCounters[kNumTimePartitions];

    RWStructuredBuffer<float> totalScattered;

    /**
     * Prefix sum to compute where a given cell (pixel) should start its indexing.
     * @param[in] pixel Pixel corresponding to a given cell.
     */
    void execute(const uint2 pixel)
    {
        if (!params.validateIntegerPixelBounds(pixel)) return;
        printSetPixel(pixel);
        print("======= Visualize Forward Reprojection =======", pixel);

        // The number of reservoirs scattered into a pixel is tracked in the cell counters.
        uint index = params.linearizePixel(pixel);

        float cellCount = 0;

        if (isMultiScatter)
        {
            float splatWeight = 1.0f / float(kNumTimePartitions);
            for (int i = 0; i < kNumTimePartitions; i++)
            {
                cellCount += splatWeight * multiCellCounters[i][index];
            }
        }
        else
        {
            cellCount = cellCounters[index];
        }

        totalScattered[index] = cellCount;
    }
};

cbuffer CB
{
    VisualizeForwardReprojection gVisualizeForwardReprojection;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID) {
    gVisualizeForwardReprojection.execute(dispatchThreadId.xy);
}
