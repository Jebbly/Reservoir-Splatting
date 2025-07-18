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
 
import Utils.Debug.PixelDebug;
import Params;
import Reservoir;

/** Compute shader that just resolves the reservoir into the final color.
*/
struct ResolveReSTIR
{
    // Resources
    PathTracerParams params;                                ///< Runtime parameters.
    StructuredBuffer<PathReservoir> currReservoirs;          ///< Input per-pixel reservoirs after all resampling passes.
    RWTexture2D<float4> outputColor;                        ///< Output resolved color.

    /** Entry point for resolve pass.
        \param[in] pixel Pixel coordinates.
    */
    void execute(const uint2 pixel)
    {
        if (any(pixel >= params.frameDim)) return;
        printSetPixel(pixel);
        print("======= Resolve ReSTIR =======", pixel);

        // Compute offset into per-sample buffers. All samples are stored consecutively at this offset.
        const uint offset = params.getSampleOffset(pixel);

        // All samples are combined into the single reservoir.
        const uint32_t reservoirIdx = params.linearizePixel(pixel);
        PathReservoir currReservoir = currReservoirs[reservoirIdx];
        float3 color = currReservoir.integrand * currReservoir.computeUCW();
        print("Integrand", currReservoir.integrand);
        print("UCW", currReservoir.computeUCW());
        print("Color", color);
        print("Final Reservoir Subpixel", currReservoir.subPixel);
        outputColor[pixel] = float4(color, 1);
    }
};

cbuffer CB
{
    ResolveReSTIR gResolveReSTIR;
}

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    gResolveReSTIR.execute(dispatchThreadId.xy);
}
