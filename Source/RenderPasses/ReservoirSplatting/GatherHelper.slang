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

import Scene.HitInfo;
import Scene.Scene;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import LoadShadingData;

import ShiftOptions;
import Params;

struct GatherData
{
    GatherMechanism gatherOption;
    Texture2D<float2> motionVectors;
    Texture2D<float2> floatingCoords;
};

ParameterBlock<GatherData> gGatherData;

struct GatherHelper
{
    int2 pixel;
    float2 prevPixel;

    __init(int2 pixel, int2 frameDim)
    {
        this.pixel = pixel;

        float2 motionVector = gGatherData.motionVectors[this.pixel];
        motionVector = (length(motionVector) < 1e-06) ? float2(0.0f, 0.0f) : motionVector;
        this.prevPixel = float2(this.pixel) + motionVector * float2(frameDim);
    }

    inline float2 getFloatingCoords()
    {
        return gGatherData.floatingCoords[this.pixel];
    }
};
