/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
#include "FrameDumper.h"
#include "Core/AssetResolver.h"

namespace
{
const char kDst[] = "dst";
const char kSrc[] = "src";

const std::string kOutputDir = "outputDir";
const std::string kOutputBase = "outputBase";
const std::string kFrameOffset = "frameOffset";
} // namespace

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, FrameDumper>();
}

FrameDumper::FrameDumper(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    for (const auto& [key, value] : props)
    {
        //if (key == kOutputDir)
        //    mOutputDir = value;
        //else if (key == kOutputBase)
        //    mOutputBase = value;
        //else
        logWarning("Unknown property '{}' in a FrameDumper properties.", key);
    }
}

RenderPassReflection FrameDumper::reflect(const CompileData& compileData)
{
    RenderPassReflection r;
    r.addOutput(kDst, "The destination texture").format(mOutputFormat);
    r.addInput(kSrc, "The source texture");
    return r;
}

Properties FrameDumper::getProperties() const
{
    Properties props;
    props[kOutputDir] = mOutputDir;
    props[kOutputBase] = mOutputBase;
    props[kFrameOffset] = mFrameOffset;
    return props;
}

void FrameDumper::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
}

void FrameDumper::createTempTexture(const ref<Texture>& inTex)
{
    mTmpTex = mpDevice->createTexture2D(
        inTex->getWidth(0),
        inTex->getHeight(0),
        ResourceFormat::RGBA8UnormSrgb,
        1,
        1,
        nullptr,
        ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource
    );
}

void FrameDumper::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pSrcTex = renderData.getTexture(kSrc);
    const auto& pDstTex = renderData.getTexture(kDst);

    // If we don't have a temp texture, make one
    if (pSrcTex && !mTmpTex)
    {
        createTempTexture(pSrcTex);
    }

    // If we have a temp texture, but it's the wrong size, recreate it.
    if (pSrcTex &&
        (mTmpTex->getWidth() != pSrcTex->getWidth() || mTmpTex->getHeight() != pSrcTex->getHeight()))
    {
        createTempTexture(pSrcTex);
    }

    if (pSrcTex && pDstTex)
    {
        if (mIsCapturing)
        {
            dumpFrame(pRenderContext, pSrcTex);
        }

        pRenderContext->blit(
            pSrcTex->getSRV(), pDstTex->getRTV(),
            RenderContext::kMaxRect, RenderContext::kMaxRect,
            TextureFilteringMode::Linear
        );
    }
    else
    {
        logWarning("FrameDumper::execute() - missing an input or output resource");
    }
}

void FrameDumper::dumpFrame(RenderContext* pRenderContext, const ref<Texture>& tex)
{
    uint32_t frameId = mFrameOffset + ( mCurFrame++ );
    char name[2048];

    if (mUsePng)
    {
        sprintf_s(name, "%s/%s.%05d.png", mOutputDir.c_str(), mOutputBase.c_str(), frameId);
        std::filesystem::path fPath = name;
        if (!mUseIntermediateTex)
        {
            tex->captureToFile(0, 0, fPath, Bitmap::FileFormat::PngFile);
        }
        else
        {
            pRenderContext->blit(tex->getSRV(), mTmpTex->getRTV());
            mTmpTex->captureToFile(0, 0, fPath, Bitmap::FileFormat::PngFile);
        }
    }
    else
    {
        sprintf_s(name, "%s/%s.%05d.pam", mOutputDir.c_str(), mOutputBase.c_str(), frameId);
        pRenderContext->blit(tex->getSRV(), mTmpTex->getRTV());
        std::vector<uint8_t> textureData = pRenderContext->readTextureSubresource(mTmpTex.get(), 0);

        FILE* fp = fopen(name, "wb");
        fprintf_s(fp, "P7\nWIDTH %d\nHEIGHT %d\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n",
            tex->getWidth(), tex->getHeight());
        fwrite(textureData.data(), sizeof(uint8_t), textureData.size(), fp);
        fclose(fp);
    }

    /*
    ref<Texture> tmpTex = mpDevice->createTexture2D(
            tex->getWidth(0),
            tex->getHeight(0),
            ResourceFormat::RGBA32Float,
            1,
            1,
            nullptr,
            ResourceBindFlags::RenderTarget | ResourceBindFlags::ShaderResource

    uint32_t subresource = tex->getSubresourceIndex(0, 0);
    std::vector<uint8_t> textureData = pRenderContext->readTextureSubresource(tex.get(), subresource);
    */
}

void FrameDumper::renderUI(Gui::Widgets& widget)
{
    if (widget.button(mIsCapturing ? "Stop Capture" : "Start Capture"))
    {
        mIsCapturing = !mIsCapturing;
        mCurFrame = 0;
    }

    widget.text("Output Directory: " + mOutputDir);
    if (widget.button("Change Output Directory"))
    {
        std::filesystem::path path;
        if (chooseFolderDialog(path))
            mOutputDir = path.string();
    }
    widget.textbox("Output Base File", mOutputBase);
    widget.var("Start Frame", mFrameOffset, 0u);
    widget.checkbox("Save PNG Files", mUsePng);
    if (mUsePng)
        widget.checkbox("   Use intermediate texture?", mUseIntermediateTex);

    if (mTmpTex)
        widget.text("Have temporary texture allocated!");
    else
        widget.text("No temporary texture available!");
}

