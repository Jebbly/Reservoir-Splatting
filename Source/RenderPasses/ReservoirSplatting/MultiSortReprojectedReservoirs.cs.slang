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

struct MultiSortReprojectedReservoirs
{
    PathTracerParams params;
    uint cellCount;

    static const uint kCounterIndexDataCount = 0;
    static const uint kCounterIndexPrefixSum = 1;
    RWStructuredBuffer<uint32_t> globalCounters[kNumTimePartitions];
    RWStructuredBuffer<uint32_t> cellCounters[kNumTimePartitions];

    RWStructuredBuffer<uint2> reservoirIndices[kNumTimePartitions];
    RWStructuredBuffer<uint2> scatteredReservoirs[kNumTimePartitions];

    RWStructuredBuffer<uint> cellOffsets[kNumTimePartitions];
    RWStructuredBuffer<uint2> sortedReservoirs[kNumTimePartitions];

    /**
     * Prefix sum to compute where a given cell (pixel) should start its indexing.
     * @param[in] pixel Pixel corresponding to a given cell.
     */
    void computeCellOffsets(const uint2 pixel)
    {
        if (!params.validateIntegerPixelBounds(pixel)) return;
        printSetPixel(pixel);
        print("======= (Multi) Compute Cell Offsets =======", pixel);

        uint index = params.linearizePixel(pixel);
        if (index >= cellCount) return;

        for (int i = 0; i < kNumTimePartitions; i++)
        {
            uint cellCounter = cellCounters[i][index];

            uint offset;
            InterlockedAdd(globalCounters[i][kCounterIndexPrefixSum], cellCounter, offset);

            // print("Cell Offset", offset);

            cellOffsets[i][index] = offset;
        }
    }

    /**
     * Use the cell offsets and the cell index to place an item into a sorted linear buffer.
     * @param[in] index Index of an item.
     */
    void sortCellData(const uint index)
    {
        for (int i = 0; i < kNumTimePartitions; i++)
        {
            if (index >= globalCounters[i][kCounterIndexDataCount]) continue; // The actual number of things that were placed in the data could be less than the cell count.

            // The index queries where the scattered reservoir from prevPixel was at.
            uint2 prevPixel = scatteredReservoirs[i][index];
            printSetPixel(prevPixel);
            print("======= (Multi) Sort Cell Data =======", prevPixel);

            const uint2 indices = reservoirIndices[i][index];
            const uint linearizedIndex = indices.x;
            const uint cellIndex = indices.y;

            // print("Sorted Index", cellOffsets[linearizedIndex] + cellIndex);

            sortedReservoirs[i][cellOffsets[i][linearizedIndex] + cellIndex] = prevPixel;
        }
    }
};

cbuffer CB
{
    MultiSortReprojectedReservoirs gMultiSortReprojectedReservoirs;
};

[shader("compute")]
[numthreads(16, 16, 1)]
void computeCellOffsets(uint3 dispatchThreadId : SV_DispatchThreadID) {
    gMultiSortReprojectedReservoirs.computeCellOffsets(dispatchThreadId.xy);
}

[shader("compute")]
[numthreads(256, 1, 1)]
void sortCellData(uint3 dispatchThreadId : SV_DispatchThreadID) {
    gMultiSortReprojectedReservoirs.sortCellData(dispatchThreadId.x);
}
