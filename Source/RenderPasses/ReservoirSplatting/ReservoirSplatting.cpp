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

#include "ReservoirSplatting.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"

namespace
{
    const uint kNumPassesWithRNG = 8; // Note: this has to be updated manually.
    const std::string kReflectTypesFile = "RenderPasses/ReservoirSplatting/ReflectTypes.cs.slang";
    // Initial Candidate Generation
    const std::string kInitialCandidatesFile = "RenderPasses/ReservoirSplatting/InitialCandidates.cs.slang";
    // Gather Temporal Resampling
    const std::string kRobustReuseOptimizationFile = "RenderPasses/ReservoirSplatting/RobustReuseOptimization.rt.slang";
    const std::string kCollectTemporalSamplesFile = "RenderPasses/ReservoirSplatting/CollectTemporalSamples.cs.slang";
    const std::string kGatherTemporalResamplingFile = "RenderPasses/ReservoirSplatting/GatherTemporalResampling.rt.slang";
    // Scatter Temporal Resampling
    const std::string kReprojectTemporalSamplesFile = "RenderPasses/ReservoirSplatting/ReprojectTemporalSamples.rt.slang";
    const std::string kSortReprojectedReservoirsFile = "RenderPasses/ReservoirSplatting/SortReprojectedReservoirs.cs.slang";
    const std::string kScatterTemporalResamplingFile = "RenderPasses/ReservoirSplatting/ScatterTemporalResampling.rt.slang";
    // Scatter + Backup Temporal Resampling
    const std::string kScatterBackupTemporalResamplingFile = "RenderPasses/ReservoirSplatting/ScatterBackupTemporalResampling.rt.slang";
    // Multi-Scatter Temporal Resampling
    const std::string kMultiReprojectTemporalSamplesFile = "RenderPasses/ReservoirSplatting/MultiReprojectTemporalSamples.rt.slang";
    const std::string kMultiSortReprojectedReservoirsFile = "RenderPasses/ReservoirSplatting/MultiSortReprojectedReservoirs.cs.slang";
    const std::string kMultiScatterTemporalResamplingFile = "RenderPasses/ReservoirSplatting/MultiScatterTemporalResampling.rt.slang";
    // Spatial Resampling
    const std::string kSpatialResamplingFile = "RenderPasses/ReservoirSplatting/SpatialResampling.rt.slang";
    // Resolve ReSTIR
    const std::string kResolveReSTIRFile = "RenderPasses/ReservoirSplatting/ResolveReSTIR.cs.slang";

    // Visualization Utility
    const std::string kVisualizeForwardReprojectionFile = "RenderPasses/ReservoirSplatting/VisualizeForwardReprojection.cs.slang";

    // Render pass inputs and outputs.

    const std::string kInputVBuffer = "vbuffer";
    const std::string kInputMotionVectors = "mvec";
    const std::string kInputViewDir = "viewW";
    const std::string kInputSampleCount = "sampleCount";

    const Falcor::ChannelList kInputChannels =
    {
        { kInputVBuffer,        "gVBuffer",         "Visibility buffer in packed format" },
        { kInputMotionVectors,  "gMotionVectors",   "Motion vector buffer (float format)"},
        // { kInputViewDir,        "gViewW",           "World-space view direction (xyz float format)", true },
        // { kInputSampleCount,    "gSampleCount",     "Sample count buffer (integer format)", true, ResourceFormat::R8Uint },
    };

    const std::string kOutputColor = "color";
    const std::string kOutputAlbedo = "albedo";
    const std::string kOutputSpecularAlbedo = "specularAlbedo";
    const std::string kOutputIndirectAlbedo = "indirectAlbedo";
    const std::string kOutputGuideNormal = "guideNormal";
    const std::string kOutputReflectionPosW = "reflectionPosW";
    const std::string kOutputRayCount = "rayCount";
    const std::string kOutputPathLength = "pathLength";

    const Falcor::ChannelList kOutputChannels =
    {
        { kOutputColor,                                     "",     "Output color (linear)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputAlbedo,                                    "",     "Output albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputSpecularAlbedo,                            "",     "Output specular albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputIndirectAlbedo,                            "",     "Output indirect albedo (linear)", true /* optional */, ResourceFormat::RGBA8Unorm },
        { kOutputGuideNormal,                               "",     "Output guide normal (linear)", true /* optional */, ResourceFormat::RGBA16Float },
        { kOutputReflectionPosW,                            "",     "Output reflection pos (world space)", true /* optional */, ResourceFormat::RGBA32Float },
        { kOutputRayCount,                                  "",     "Per-pixel ray count", true /* optional */, ResourceFormat::R32Uint },
        { kOutputPathLength,                                "",     "Per-pixel path length", true /* optional */, ResourceFormat::R32Uint },
    };

    // Scripting options.
    const std::string kSamplesPerPixel = "samplesPerPixel";
    const std::string kMaxSurfaceBounces = "maxSurfaceBounces";
    const std::string kMaxDiffuseBounces = "maxDiffuseBounces";
    const std::string kMaxSpecularBounces = "maxSpecularBounces";
    const std::string kMaxTransmissionBounces = "maxTransmissionBounces";

    const std::string kSampleGenerator = "sampleGenerator";
    const std::string kFixedSeed = "fixedSeed";
    const std::string kUseRussianRoulette = "useRussianRoulette";
    const std::string kMISHeuristic = "misHeuristic";
    const std::string kMISPowerExponent = "misPowerExponent";
    const std::string kEmissiveSampler = "emissiveSampler";
    const std::string kLightBVHOptions = "lightBVHOptions";

    const std::string kUseAlphaTest = "useAlphaTest";
    const std::string kAdjustShadingNormals = "adjustShadingNormals";
    const std::string kMaxNestedMaterials = "maxNestedMaterials";
    const std::string kUseLightsInDielectricVolumes = "useLightsInDielectricVolumes";
    const std::string kDisableCaustics = "disableCaustics";
    const std::string kSpecularRoughnessThreshold = "specularRoughnessThreshold";
    const std::string kPrimaryLodMode = "primaryLodMode";
    const std::string kLODBias = "lodBias";

    const std::string kOutputSize = "outputSize";
    const std::string kFixedOutputSize = "fixedOutputSize";
    const std::string kColorFormat = "colorFormat";

    const std::string kEnableTemporalResampling = "enableTemporalResampling";
    const std::string kEnableSpatialResampling = "enableSpatialResampling";
    const std::string kTemporalReuse = "temporalReuse";
    const std::string kScatterBackupMISOption = "scatterBackupMISOption";
    const std::string kGatherOption = "gatherOption";
    const std::string kNumTimePartitions = "numTimePartitions";

    const uint32_t kSpatialNeighborSamples = 8192;
    }

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReservoirSplatting>();
    ScriptBindings::registerBinding(ReservoirSplatting::registerBindings);
}

void ReservoirSplatting::registerBindings(pybind11::module& m)
{
    pybind11::class_<ReservoirSplatting, RenderPass, ref<ReservoirSplatting>> pass(m, "ReservoirSplatting");
    pass.def("reset", &ReservoirSplatting::reset);
    pass.def_property_readonly("pixelStats", &ReservoirSplatting::getPixelStats);

    pass.def_property("useFixedSeed",
        [](const ReservoirSplatting* pt) { return pt->mParams.useFixedSeed ? true : false; },
        [](ReservoirSplatting* pt, bool value) { pt->mParams.useFixedSeed = value ? 1 : 0; }
    );
    pass.def_property("fixedSeed",
        [](const ReservoirSplatting* pt) { return pt->mParams.fixedSeed; },
        [](ReservoirSplatting* pt, uint32_t value) { pt->mParams.fixedSeed = value; }
    );
}

ReservoirSplatting::ReservoirSplatting(ref<Device> pDevice, const Properties& props)
    : RenderPass(pDevice)
{
    if (!mpDevice->isShaderModelSupported(ShaderModel::SM6_5))
        FALCOR_THROW("ReservoirSplatting requires Shader Model 6.5 support.");
    if (!mpDevice->isFeatureSupported(Device::SupportedFeatures::RaytracingTier1_1))
        FALCOR_THROW("ReservoirSplatting requires Raytracing Tier 1.1 support.");

    parseProperties(props);
    validateOptions();

    // Create sample generator.
    mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);

    // Create resolve pass. This doesn't depend on the scene so can be created here.
    auto defines = mStaticParams.getDefines(*this);
    mpResolveReSTIRPass = ComputePass::create(mpDevice, ProgramDesc().addShaderLibrary(kResolveReSTIRFile).csEntry("main"), defines, false);

    // Note: The other programs are lazily created in updatePrograms() because a scene needs to be present when creating them.

    mpPixelStats = std::make_unique<PixelStats>(mpDevice);
    mpPixelDebug = std::make_unique<PixelDebug>(mpDevice);
}

void ReservoirSplatting::setProperties(const Properties& props)
{
    parseProperties(props);
    validateOptions();
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
        lightBVHSampler->setOptions(mLightBVHOptions);
    mRecompile = true;
    mOptionsChanged = true;
}

void ReservoirSplatting::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        // Rendering parameters
        if (key == kSamplesPerPixel) mStaticParams.samplesPerPixel = value;
        else if (key == kMaxSurfaceBounces) mStaticParams.maxSurfaceBounces = value;
        else if (key == kMaxDiffuseBounces) mStaticParams.maxDiffuseBounces = value;
        else if (key == kMaxSpecularBounces) mStaticParams.maxSpecularBounces = value;
        else if (key == kMaxTransmissionBounces) mStaticParams.maxTransmissionBounces = value;

        // Sampling parameters
        else if (key == kSampleGenerator) mStaticParams.sampleGenerator = value;
        else if (key == kFixedSeed) { mParams.fixedSeed = value; mParams.useFixedSeed = true; }
        else if (key == kUseRussianRoulette) mStaticParams.useRussianRoulette = value;
        else if (key == kMISHeuristic) mStaticParams.misHeuristic = value;
        else if (key == kMISPowerExponent) mStaticParams.misPowerExponent = value;
        else if (key == kEmissiveSampler) mStaticParams.emissiveSampler = value;
        else if (key == kLightBVHOptions) mLightBVHOptions = value;

        // Material parameters
        else if (key == kUseAlphaTest) mStaticParams.useAlphaTest = value;
        else if (key == kAdjustShadingNormals) mStaticParams.adjustShadingNormals = value;
        else if (key == kMaxNestedMaterials) mStaticParams.maxNestedMaterials = value;
        else if (key == kUseLightsInDielectricVolumes) mStaticParams.useLightsInDielectricVolumes = value;
        else if (key == kDisableCaustics) mStaticParams.disableCaustics = value;
        else if (key == kSpecularRoughnessThreshold) mParams.specularRoughnessThreshold = value;
        else if (key == kPrimaryLodMode) mStaticParams.primaryLodMode = value;
        else if (key == kLODBias) mParams.lodBias = value;

        // Output parameters
        else if (key == kOutputSize) mOutputSizeSelection = value;
        else if (key == kFixedOutputSize) mFixedOutputSize = value;
        else if (key == kColorFormat) mStaticParams.colorFormat = value;

        // ReSTIR parameters
        else if (key == kEnableTemporalResampling) mReSTIRParams.enableTemporalResampling = value;
        else if (key == kEnableSpatialResampling) mReSTIRParams.enableSpatialResampling = value;
        else if (key == kTemporalReuse) mReSTIRParams.temporalReuseOption = value;
        else if (key == kScatterBackupMISOption) mReSTIRParams.scatterBackupMISOption = value;
        else if (key == kGatherOption) mReSTIRParams.temporalGathering = value;
        else if (key == kNumTimePartitions) mReSTIRParams.numTimePartitions = value;

        else logWarning("Unknown property '{}' in ReservoirSplatting properties.", key);
    }

    if (props.has(kMaxSurfaceBounces))
    {
        // Initialize bounce counts to 'maxSurfaceBounces' if they weren't explicitly set.
        if (!props.has(kMaxDiffuseBounces)) mStaticParams.maxDiffuseBounces = mStaticParams.maxSurfaceBounces;
        if (!props.has(kMaxSpecularBounces)) mStaticParams.maxSpecularBounces = mStaticParams.maxSurfaceBounces;
        if (!props.has(kMaxTransmissionBounces)) mStaticParams.maxTransmissionBounces = mStaticParams.maxSurfaceBounces;
    }
    else
    {
        // Initialize surface bounces.
        mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    }

    bool maxSurfaceBouncesNeedsAdjustment =
        mStaticParams.maxSurfaceBounces < mStaticParams.maxDiffuseBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxSpecularBounces ||
        mStaticParams.maxSurfaceBounces < mStaticParams.maxTransmissionBounces;

    // Show a warning if maxSurfaceBounces will be adjusted in validateOptions().
    if (props.has(kMaxSurfaceBounces) && maxSurfaceBouncesNeedsAdjustment)
    {
        logWarning("'{}' is set lower than '{}', '{}' or '{}' and will be increased.", kMaxSurfaceBounces, kMaxDiffuseBounces, kMaxSpecularBounces, kMaxTransmissionBounces);
    }
}

void ReservoirSplatting::validateOptions()
{
    if (mParams.specularRoughnessThreshold < 0.f || mParams.specularRoughnessThreshold > 1.f)
    {
        logWarning("'specularRoughnessThreshold' has invalid value. Clamping to range [0,1].");
        mParams.specularRoughnessThreshold = std::clamp(mParams.specularRoughnessThreshold, 0.f, 1.f);
    }

    // Static parameters.
    if (mStaticParams.samplesPerPixel < 1 || mStaticParams.samplesPerPixel > kMaxSamplesPerPixel)
    {
        logWarning("'samplesPerPixel' must be in the range [1, {}]. Clamping to this range.", kMaxSamplesPerPixel);
        mStaticParams.samplesPerPixel = std::clamp(mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);
    }

    auto clampBounces = [] (uint32_t& bounces, const std::string& name)
    {
        if (bounces > kMaxBounces)
        {
            logWarning("'{}' exceeds the maximum supported bounces. Clamping to {}.", name, kMaxBounces);
            bounces = kMaxBounces;
        }
    };

    clampBounces(mStaticParams.maxSurfaceBounces, kMaxSurfaceBounces);
    clampBounces(mStaticParams.maxDiffuseBounces, kMaxDiffuseBounces);
    clampBounces(mStaticParams.maxSpecularBounces, kMaxSpecularBounces);
    clampBounces(mStaticParams.maxTransmissionBounces, kMaxTransmissionBounces);

    // Make sure maxSurfaceBounces is at least as many as any of diffuse, specular or transmission.
    uint32_t minSurfaceBounces = std::max(mStaticParams.maxDiffuseBounces, std::max(mStaticParams.maxSpecularBounces, mStaticParams.maxTransmissionBounces));
    mStaticParams.maxSurfaceBounces = std::max(mStaticParams.maxSurfaceBounces, minSurfaceBounces);

    if (mStaticParams.primaryLodMode == TexLODMode::RayCones)
    {
        logWarning("Unsupported tex lod mode. Defaulting to Mip0.");
        mStaticParams.primaryLodMode = TexLODMode::Mip0;
    }
}

Properties ReservoirSplatting::getProperties() const
{
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    Properties props;

    // Rendering parameters
    props[kSamplesPerPixel] = mStaticParams.samplesPerPixel;
    props[kMaxSurfaceBounces] = mStaticParams.maxSurfaceBounces;
    props[kMaxDiffuseBounces] = mStaticParams.maxDiffuseBounces;
    props[kMaxSpecularBounces] = mStaticParams.maxSpecularBounces;
    props[kMaxTransmissionBounces] = mStaticParams.maxTransmissionBounces;

    // Sampling parameters
    props[kSampleGenerator] = mStaticParams.sampleGenerator;
    if (mParams.useFixedSeed) props[kFixedSeed] = mParams.fixedSeed;
    props[kUseRussianRoulette] = mStaticParams.useRussianRoulette;
    props[kMISHeuristic] = mStaticParams.misHeuristic;
    props[kMISPowerExponent] = mStaticParams.misPowerExponent;
    props[kEmissiveSampler] = mStaticParams.emissiveSampler;
    if (mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH) props[kLightBVHOptions] = mLightBVHOptions;

    // Material parameters
    props[kUseAlphaTest] = mStaticParams.useAlphaTest;
    props[kAdjustShadingNormals] = mStaticParams.adjustShadingNormals;
    props[kMaxNestedMaterials] = mStaticParams.maxNestedMaterials;
    props[kUseLightsInDielectricVolumes] = mStaticParams.useLightsInDielectricVolumes;
    props[kDisableCaustics] = mStaticParams.disableCaustics;
    props[kSpecularRoughnessThreshold] = mParams.specularRoughnessThreshold;
    props[kPrimaryLodMode] = mStaticParams.primaryLodMode;
    props[kLODBias] = mParams.lodBias;

    // Output parameters
    props[kOutputSize] = mOutputSizeSelection;
    if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed) props[kFixedOutputSize] = mFixedOutputSize;
    props[kColorFormat] = mStaticParams.colorFormat;

    return props;
}

RenderPassReflection ReservoirSplatting::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    const uint2 sz = RenderPassHelpers::calculateIOSize(mOutputSizeSelection, mFixedOutputSize, compileData.defaultTexDims);

    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels, ResourceBindFlags::UnorderedAccess, sz);
    return reflector;
}

void ReservoirSplatting::setFrameDim(const uint2 frameDim)
{
    auto prevFrameDim = mParams.frameDim;
    auto prevScreenTiles = mParams.screenTiles;

    mParams.frameDim = frameDim;
    if (mParams.frameDim.x > kMaxFrameDimension || mParams.frameDim.y > kMaxFrameDimension)
    {
        FALCOR_THROW("Frame dimensions up to {} pixels width/height are supported.", kMaxFrameDimension);
    }

    // Tile dimensions have to be powers-of-two.
    FALCOR_ASSERT(isPowerOf2(kScreenTileDim.x) && isPowerOf2(kScreenTileDim.y));
    FALCOR_ASSERT(kScreenTileDim.x == (1 << kScreenTileBits.x) && kScreenTileDim.y == (1 << kScreenTileBits.y));
    mParams.screenTiles = div_round_up(mParams.frameDim, kScreenTileDim);

    if (any(mParams.frameDim != prevFrameDim) || any(mParams.screenTiles != prevScreenTiles))
    {
        mVarsChanged = true;
    }
}

void ReservoirSplatting::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mParams.frameCount = 0;
    mParams.frameDim = {};
    mParams.screenTiles = {};

    resetPrograms();
    resetLighting();

    if (mpScene)
    {
        if (pScene->hasGeometryType(Scene::GeometryType::Custom))
        {
            logWarning("ReservoirSplatting: This render pass does not support custom primitives.");
        }

        validateOptions();
    }
}

void ReservoirSplatting::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    if (!beginFrame(pRenderContext, renderData))
        return;

    bool shouldReset = mpScene->shouldReset();

    // Update shader program specialization.
    updatePrograms();

    // Prepare resources.
    prepareResources(pRenderContext, renderData);

    // Prepare the path tracer and camera manager parameter block.
    // This should be called after all resources have been created.
    preparePathTracer(renderData);
    prepareCameraManager(renderData, shouldReset);
    prepareGatherData(pRenderContext, renderData);
    mVarsChanged = false;

    // Initial candidate generation for canonical samples.
    initialCandidateGeneration(pRenderContext, renderData);

    // Temporal resampling after initial candidate generation.
    shouldReset |= (mParams.frameCount == 0);
    if (mReSTIRParams.enableTemporalResampling && !shouldReset)
    {
        switch (mReSTIRParams.temporalReuseOption)
        {
            case TemporalReuse::GatherOnly:
            {
                if (mReSTIRParams.temporalGathering == GatherMechanism::Robust)
                {
                    robustReuseOptimization(pRenderContext, renderData);
                }
                collectTemporalSamples(pRenderContext, renderData);
                gatherTemporalResampling(pRenderContext, renderData);
                break;
            }
            case TemporalReuse::ScatterOnly:
            {
                reprojectTemporalSamples(pRenderContext, renderData);
                sortReprojectedReservoirs(pRenderContext, renderData);
                scatterTemporalResampling(pRenderContext, renderData);
                break;
            }
            case TemporalReuse::ScatterBackup:
            {
                if (mReSTIRParams.temporalGathering == GatherMechanism::Robust)
                {
                    robustReuseOptimization(pRenderContext, renderData);
                }
                collectTemporalSamples(pRenderContext, renderData);
                reprojectTemporalSamples(pRenderContext, renderData);
                sortReprojectedReservoirs(pRenderContext, renderData);
                scatterBackupTemporalResampling(pRenderContext, renderData);
                break;
            }
            case TemporalReuse::MultiScatter:
            {
                multiReprojectTemporalSamples(pRenderContext, renderData);
                multiSortReprojectedReservoirs(pRenderContext, renderData);
                multiScatterTemporalResampling(pRenderContext, renderData);
                break;
            }
        }
    }


    if (mDumpScatterCount)
    {
        visualizeForwardReprojection(pRenderContext, renderData);

        const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;

        pRenderContext->copyResource(mpStagingTotalCellCounters.get(), mpTotalCellCounters.get());
        pRenderContext->submit(false);
        pRenderContext->signal(mpReadbackFence.get());
        mpReadbackFence->wait();
        const float* pTotalScattered = reinterpret_cast<const float*>(mpStagingTotalCellCounters->map());
        float* totalScattered = new float[screenPixelCount];
        std::memcpy(totalScattered, pTotalScattered, sizeof(float) * screenPixelCount);

        std::ofstream output(mScatterDumpDir + "/" + mScatterDumpFile);
        for (uint32_t i = 0; i < screenPixelCount; i++)
        {
            output << totalScattered[i] << "\n";
        }
        output.close();

        delete[] totalScattered;
        mpStagingTotalCellCounters->unmap();

        mDumpScatterCount = false;
    }

    if (mReSTIRParams.enableSpatialResampling)
    {
        spatialResampling(pRenderContext, renderData);
    }

    // Resolve pass.
    resolveReSTIR(pRenderContext, renderData);

    auto& dict = renderData.getDictionary();
    dict[Falcor::kSuppressAutoReset] = mpScene->isFrozen() || mpScene->isReplaying();
    dict[Falcor::kAccumulatePassShouldRun] = mpScene->shouldAccumulate();
    dict[Falcor::kViewAccumulatingPass] = mpScene->viewAccumulation();

    endFrame(pRenderContext, renderData);
}

void ReservoirSplatting::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;

    // ReSTIR options.
    if (auto restirGroup = widget.group("ReSTIR Options", false))
    {
        dirty |= renderReSTIRUI(restirGroup);
    }

    // Rendering options.
    if (auto renderingGroup = widget.group("Rendering Options", false))
    {
        dirty |= renderRenderingUI(renderingGroup);
    }

    // Stats and debug options.
    if (auto statGroup = widget.group("Statistics", false))
    {
        renderStatsUI(statGroup);
    }

    if (auto debugGroup = widget.group("Debugging", false))
    {
        dirty |= renderDebugUI(debugGroup);
    }

    if (dirty)
    {
        validateOptions();
        mOptionsChanged = true;
    }
}

bool ReservoirSplatting::renderScatterVisualizationUI(Gui::Widgets& widget)
{
    bool dirty = false;

    widget.text("Output Directory: " + mScatterDumpDir);
    if (widget.button("Change Output Directory"))
    {
        std::filesystem::path path;
        if (chooseFolderDialog(path))
            mScatterDumpDir = path.string();
    }
    widget.textbox("Output Text File", mScatterDumpFile);

    if (widget.button("Dump Scatter Counts"))
        mDumpScatterCount = true;

    return dirty;
}

bool ReservoirSplatting::renderReSTIRUI(Gui::Widgets& widget)
{
    bool dirty = false;

    if (auto temporal = widget.group("Temporal Resampling", false))
    {
        dirty |= temporal.checkbox("Enable Temporal Resampling", mReSTIRParams.enableTemporalResampling);
        temporal.tooltip("Enables temporal resampling.", true);

        if (mReSTIRParams.enableTemporalResampling)
        {
            const bool noMotionBlur = (mpScene == nullptr) || (mpScene->getCamera()->getShutterSpeed() == 0.0);

            // Make sure to delete the multi-splatting option if motion blur is disabled.
            std::function<bool(TemporalReuse)> filter = [noMotionBlur](TemporalReuse reuseOption) {
                return !(noMotionBlur && reuseOption == TemporalReuse::MultiScatter);
            };

            dirty |= temporal.dropdown("Temporal Reuse Option", mReSTIRParams.temporalReuseOption, false, filter);
            temporal.tooltip("Reuse option for temporal resampling.", true);

            // There is a case where multi-splat was selected but the shutter speed is set to 0.
            // In this case, we should default back to the scatter only option.
            if (noMotionBlur && mReSTIRParams.temporalReuseOption == TemporalReuse::MultiScatter)
            {
                dirty = true;
                mReSTIRParams.temporalReuseOption = TemporalReuse::ScatterOnly;
            }

            if (mReSTIRParams.temporalReuseOption == TemporalReuse::GatherOnly ||
                mReSTIRParams.temporalReuseOption == TemporalReuse::ScatterBackup)
            {
                dirty |= temporal.dropdown("Gather Mechanism", mReSTIRParams.temporalGathering);
                temporal.tooltip("Gather option for temporal resamplling.", true);
            }

            if (mReSTIRParams.temporalReuseOption == TemporalReuse::ScatterOnly||
                mReSTIRParams.temporalReuseOption == TemporalReuse::MultiScatter ||
                mReSTIRParams.temporalReuseOption == TemporalReuse::ScatterBackup)
            {
                if (mReSTIRParams.temporalReuseOption == TemporalReuse::MultiScatter)
                {
                    mRecompile |= temporal.var("Time Partitions", mReSTIRParams.numTimePartitions, uint32_t(1));
                    temporal.tooltip("Number of partitions to scatter into.", true);
                }

                if (auto visualizationGroup = widget.group("Scatter Visualization", false))
                {
                    dirty |= renderScatterVisualizationUI(visualizationGroup);
                }
            }

            if (mReSTIRParams.temporalReuseOption == TemporalReuse::ScatterBackup)
            {
                dirty |= temporal.dropdown("Scatter Backup MIS Weighting Scheme", mReSTIRParams.scatterBackupMISOption);
                temporal.tooltip("MIS weighting between scatter and gather options.");
            }
            dirty |= mRecompile;

            dirty |= temporal.checkbox("Enable Confidence Weights Temporally", mReSTIRParams.useConfidenceWeightsTemporally);
            temporal.tooltip("Use confidence weights during temporal resampling.", true);
        }
    }

    if (auto spatial = widget.group("Spatial Resampling", false))
    {
        dirty |= spatial.checkbox("Enable Spatial Resampling", mReSTIRParams.enableSpatialResampling);
        spatial.tooltip("Enables spatial resampling.", true);

        if (mReSTIRParams.enableSpatialResampling)
        {
            dirty |= spatial.var("Spatial Resampling Iterations", mReSTIRParams.numSpatialIterations);
            spatial.tooltip("Number of iterations to repeat spatial resampling", true);

            dirty |= spatial.var("Spatial Gather Radius", mReSTIRParams.spatialGatherRadius);
            spatial.tooltip("Radius of disk for candidate spatial samples", true);

            dirty |= spatial.var("Spatial Neighbor Samples", mReSTIRParams.neighborCount);
            spatial.tooltip("Number of spatial neighbors to resample per iteration.", true);

            dirty |= spatial.checkbox("Enable Confidence Weights Spatially", mReSTIRParams.useConfidenceWeightsSpatially);
            spatial.tooltip("Use confidence weights during spatial resampling.", true);
        }
    }

    if (dirty)
    {
        reset();
    }

    return dirty;
}

bool ReservoirSplatting::renderRenderingUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool runtimeDirty = false;

    dirty |= widget.var("Simulated Frame Time", mCameraParams.artificialFrameTime, 0.01f, 10000.0f);
    widget.tooltip("Artificially defined frame time between the camera frames.", true);

    dirty |= widget.var("Samples / Pixel", mStaticParams.samplesPerPixel, 1u, kMaxSamplesPerPixel);

    if (widget.var("Max Surface Bounces", mStaticParams.maxSurfaceBounces, 0u, kMaxBounces))
    {
        // Allow users to change the max surface bounce parameter in the UI to clamp all other surface bounce parameters.
        mStaticParams.maxDiffuseBounces = std::min(mStaticParams.maxDiffuseBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxSpecularBounces = std::min(mStaticParams.maxSpecularBounces, mStaticParams.maxSurfaceBounces);
        mStaticParams.maxTransmissionBounces = std::min(mStaticParams.maxTransmissionBounces, mStaticParams.maxSurfaceBounces);
        dirty = true;
    }
    widget.tooltip("Maximum number of surface bounces (diffuse + specular + transmission).\n"
        "Note that specular reflection events from a material with a roughness greater than specularRoughnessThreshold are also classified as diffuse events.");

    dirty |= widget.var("Max Diffuse Bounces", mStaticParams.maxDiffuseBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of diffuse bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max Specular Bounces", mStaticParams.maxSpecularBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of specular bounces.\n0 = direct only\n1 = one indirect bounce etc.");

    dirty |= widget.var("Max Transmission Bounces", mStaticParams.maxTransmissionBounces, 0u, kMaxBounces);
    widget.tooltip("Maximum number of transmission bounces.\n0 = no transmission\n1 = one transmission bounce etc.");

    // Sampling options.

    if (widget.dropdown("Sample Generator", SampleGenerator::getGuiDropdownList(), mStaticParams.sampleGenerator))
    {
        mpSampleGenerator = SampleGenerator::create(mpDevice, mStaticParams.sampleGenerator);
        dirty = true;
    }

    dirty |= widget.checkbox("Russian Roulette", mStaticParams.useRussianRoulette);
    widget.tooltip("Use russian roulette to terminate low throughput paths.");

    dirty |= widget.dropdown("MIS Heuristic", mStaticParams.misHeuristic);

    if (mStaticParams.misHeuristic == MISHeuristic::PowerExp)
    {
        dirty |= widget.var("MIS Power Exponent", mStaticParams.misPowerExponent, 0.01f, 10.f);
    }

    if (mpScene && mpScene->useEmissiveLights())
    {
        if (auto group = widget.group("Emissive Sampler"))
        {
            // LightBVHs cannot be used with ReSTIR.
            bool enabledReSTIR = mReSTIRParams.enableTemporalResampling || mReSTIRParams.enableSpatialResampling;
            std::function<bool(EmissiveLightSamplerType)> filter = [enabledReSTIR](EmissiveLightSamplerType lightSampler)
            { return !(enabledReSTIR && lightSampler == EmissiveLightSamplerType::LightBVH) &&
                      (lightSampler != EmissiveLightSamplerType::Null); };

            if (widget.dropdown("Emissive Sampler", mStaticParams.emissiveSampler, false, filter))
            {
                resetLighting();
                dirty = true;
            }
            widget.tooltip("Selects which light sampler to use for importance sampling of emissive geometry.", true);

            // There is a case where the LightBVH was selected before ReSTIR was enabled.
            // In this case, we should default back to the power emissive light sampler.
            if (enabledReSTIR && mStaticParams.emissiveSampler == EmissiveLightSamplerType::LightBVH)
            {
                resetLighting();
                dirty = true;
                mStaticParams.emissiveSampler = EmissiveLightSamplerType::Power;
            }

            if (mpEmissiveSampler)
            {
                if (mpEmissiveSampler->renderUI(group)) mOptionsChanged = true;
            }
        }
    }

    if (auto group = widget.group("Material Controls"))
    {
        dirty |= widget.checkbox("Alpha Test", mStaticParams.useAlphaTest);
        widget.tooltip("Use alpha testing on non-opaque triangles.");

        dirty |= widget.checkbox("Adjust Secondary Hit Shading Normals", mStaticParams.adjustShadingNormals);
        widget.tooltip("Enables adjustment of the shading normals to reduce the risk of black pixels due to back-facing vectors.\nDoes not apply to primary hits which is configured in GBuffer.", true);

        dirty |= widget.var("Max Nested Materials", mStaticParams.maxNestedMaterials, 2u, 4u);
        widget.tooltip("Maximum supported number of nested materials.");

        dirty |= widget.checkbox("Use Lights in Dielectric Volumes", mStaticParams.useLightsInDielectricVolumes);
        widget.tooltip("Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.");

        dirty |= widget.checkbox("Disable Caustics", mStaticParams.disableCaustics);
        widget.tooltip("Disable sampling of caustic light paths (i.e. specular events after diffuse events).");

        runtimeDirty |= widget.var("Specular Roughness Threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Specular reflection events are only classified as specular if the material's roughness value is equal or smaller than this threshold. Otherwise they are classified diffuse.");

        dirty |= widget.dropdown("Primary LOD Mode", mStaticParams.primaryLodMode);
        widget.tooltip("Texture LOD mode at primary hit");

        runtimeDirty |= widget.var("TexLOD Bias", mParams.lodBias, -16.f, 16.f, 0.01f);
    }

    if (auto group = widget.group("Output Options"))
    {
        // Switch to enable/disable path tracer output.
        dirty |= widget.checkbox("Enable Output", mEnabled);

        // Controls for output size.
        // When output size requirements change, we'll trigger a graph recompile to update the render pass I/O sizes.
        if (widget.dropdown("Output Size", mOutputSizeSelection)) requestRecompile();
        if (mOutputSizeSelection == RenderPassHelpers::IOSize::Fixed)
        {
            if (widget.var("Size in Pixels", mFixedOutputSize, 32u, 16384u)) requestRecompile();
        }

        dirty |= widget.dropdown("Color Format", mStaticParams.colorFormat);
        widget.tooltip("Selects the color format used for internal per-sample color and denoiser buffers");
    }

    if (dirty) mRecompile = true;
    return dirty || runtimeDirty;
}

bool ReservoirSplatting::renderDebugUI(Gui::Widgets& widget)
{
    bool dirty = false;

    dirty |= widget.checkbox("Use Fixed Seed", mParams.useFixedSeed);
    widget.tooltip("Forces a fixed random seed for each frame.\n\n"
        "This should produce exactly the same image each frame, which can be useful for debugging. Note that this gets ignored for now.");
    if (mParams.useFixedSeed)
    {
        dirty |= widget.var("Seed", mParams.fixedSeed);
    }

    mpPixelDebug->renderUI(widget);

    return dirty;
}

void ReservoirSplatting::renderStatsUI(Gui::Widgets& widget)
{
    // Show ray stats
    mpPixelStats->renderUI(widget);
}

bool ReservoirSplatting::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpPixelDebug->onMouseEvent(mouseEvent);
}

void ReservoirSplatting::reset()
{
    if (mpScene && !mpScene->isFrozen())
    {
        mParams.frameCount = 0;
        mParams.seed = 0;
    }
}

ReservoirSplatting::TracePass::TracePass(ref<Device> pDevice, const std::string& filename, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances)
    : name(name)
    , passDefine(passDefine)
{
    const uint32_t kRayTypeScatter = 0;
    const uint32_t kRayTypeShadow = 1;
    const uint32_t kMissScatter = 0;
    const uint32_t kMissShadow = 1;

    ProgramDesc desc;
    desc.addShaderModules(pScene->getShaderModules());
    desc.addShaderLibrary(filename);
    desc.setMaxPayloadSize(500); // This is conservative but the required minimum is 140 bytes.
    desc.setMaxAttributeSize(pScene->getRaytracingMaxAttributeSize());
    desc.setMaxTraceRecursionDepth(1);
    if (!pScene->hasProceduralGeometry()) desc.setRtPipelineFlags(RtPipelineFlags::SkipProceduralPrimitives);

    // Create ray tracing binding table.
    pBindingTable = RtBindingTable::create(2, 2, pScene->getGeometryCount());

    // Specify entry point for raygen and miss shaders.
    // The raygen shader needs type conformances for *all* materials in the scene.
    // The miss shader doesn't need need any type conformances because it does not use materials.
    pBindingTable->setRayGen(desc.addRayGen("rayGen", globalTypeConformances));
    pBindingTable->setMiss(kMissScatter, desc.addMiss("scatterMiss"));
    pBindingTable->setMiss(kMissShadow, desc.addMiss("shadowMiss"));

    // Specify hit group entry points for every combination of geometry and material type.
    // The code for each hit group gets specialized for the actual types it's operating on.
    // First query which material types the scene has.
    auto materialTypes = pScene->getMaterialSystem().getMaterialTypes();

    for (const auto materialType : materialTypes)
    {
        auto typeConformances = pScene->getMaterialSystem().getTypeConformances(materialType);

        // Add hit groups for triangles.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::TriangleMesh, materialType); !geometryIDs.empty())
        {
            auto scatterShaderID = desc.addHitGroup("scatterTriangleClosestHit", "scatterTriangleAnyHit", "", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, scatterShaderID);

            auto shadowShaderID =desc.addHitGroup("", "shadowTriangleAnyHit", "", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeShadow, geometryIDs, shadowShaderID);
        }

        // Add hit groups for displaced triangle meshes.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::DisplacedTriangleMesh, materialType); !geometryIDs.empty())
        {
            auto scatterShaderID = desc.addHitGroup("scatterDisplacedTriangleMeshClosestHit", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, scatterShaderID);

            auto shadowShaderID = desc.addHitGroup("", "", "displacedTriangleMeshIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeShadow, geometryIDs, shadowShaderID);
        }

        // Add hit groups for curves.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::Curve, materialType); !geometryIDs.empty())
        {
            auto scatterShaderID = desc.addHitGroup("scatterCurveClosestHit", "", "curveIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, scatterShaderID);

            auto shadowShaderID = desc.addHitGroup("", "", "curveIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeShadow, geometryIDs, shadowShaderID);
        }

        // Add hit groups for SDF grids.
        if (auto geometryIDs = pScene->getGeometryIDs(Scene::GeometryType::SDFGrid, materialType); !geometryIDs.empty())
        {
            auto scatterShaderID = desc.addHitGroup("scatterSdfGridClosestHit", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeScatter, geometryIDs, scatterShaderID);

            auto shadowShaderID = desc.addHitGroup("", "", "sdfGridIntersection", typeConformances, to_string(materialType));
            pBindingTable->setHitGroup(kRayTypeShadow, geometryIDs, shadowShaderID);
        }
    }

    pProgram = Program::create(pDevice, desc, defines);
}

void ReservoirSplatting::TracePass::prepareProgram(ref<Device> pDevice, const DefineList& defines)
{
    FALCOR_ASSERT(pProgram != nullptr && pBindingTable != nullptr);
    pProgram->setDefines(defines);
    if (!passDefine.empty()) pProgram->addDefine(passDefine);
    pVars = RtProgramVars::create(pDevice, pProgram, pBindingTable);
}

void ReservoirSplatting::resetPrograms()
{
    mpInitialCandidatesPass = nullptr;
    mpRobustReuseOptimizationPass = nullptr;
    mpCollectTemporalSamplesPass = nullptr;
    mpReprojectTemporalSamplesPass = nullptr;
    mpMultiReprojectTemporalSamplesPass = nullptr;
    mpComputeCellOffsetsPass = nullptr;
    mpSortCellDataPass = nullptr;
    mpMultiComputeCellOffsetsPass = nullptr;
    mpMultiSortCellDataPass = nullptr;
    mpVisualizeForwardReprojectionPass = nullptr;
    mpGatherTemporalResamplingPass = nullptr;
    mpScatterTemporalResamplingPass = nullptr;
    mpScatterBackupTemporalResamplingPass = nullptr;
    mpMultiScatterTemporalResamplingPass = nullptr;
    mpSpatialResamplingPass = nullptr;

    mpReflectTypes = nullptr;

    mRecompile = true;
}

void ReservoirSplatting::updatePrograms()
{
    FALCOR_ASSERT(mpScene);

    if (mRecompile == false)
        return;

    // If we get here, a change that require recompilation of shader programs has occurred.
    // This may be due to change of scene defines, type conformances, shader modules, or other changes that require recompilation.
    // When type conformances and/or shader modules change, the programs need to be recreated. We assume programs have been reset upon such
    // changes. When only defines have changed, it is sufficient to update the existing programs and recreate the program vars.

    auto defines = mStaticParams.getDefines(*this);
    if (mpEmissiveSampler)
        defines.add(mpEmissiveSampler->getDefines());

    auto globalTypeConformances = mpScene->getTypeConformances();

    // Create ReSTIR passes.
    ProgramDesc baseDesc;
    baseDesc.addShaderModules(mpScene->getShaderModules());
    baseDesc.addTypeConformances(globalTypeConformances);

    // if (!mpInitialCandidatesPass)
    //     mpInitialCandidatesPass = std::make_unique<TracePass>(mpDevice, kInitialCandidatesFile, "initialCandidates", "", mpScene, defines, globalTypeConformances);
    // mpInitialCandidatesPass->prepareProgram(mpDevice, defines);

    if (!mpInitialCandidatesPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kInitialCandidatesFile).csEntry("main");
        mpInitialCandidatesPass = ComputePass::create(mpDevice, desc, defines, false);
    }
    mpInitialCandidatesPass->getProgram()->addDefines(defines);
    mpInitialCandidatesPass->setVars(nullptr);

    if (!mpRobustReuseOptimizationPass)
        mpRobustReuseOptimizationPass = std::make_unique<TracePass>(mpDevice, kRobustReuseOptimizationFile, "robustReuseOptimization", "", mpScene, defines, globalTypeConformances);
    mpRobustReuseOptimizationPass->prepareProgram(mpDevice, defines);

    if (!mpCollectTemporalSamplesPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kCollectTemporalSamplesFile).csEntry("main");
        mpCollectTemporalSamplesPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpReprojectTemporalSamplesPass)
        mpReprojectTemporalSamplesPass = std::make_unique<TracePass>(mpDevice, kReprojectTemporalSamplesFile, "reprojectTemporalSamples", "", mpScene, defines, globalTypeConformances);
    mpReprojectTemporalSamplesPass->prepareProgram(mpDevice, defines);

    if (!mpMultiReprojectTemporalSamplesPass)
        mpMultiReprojectTemporalSamplesPass = std::make_unique<TracePass>(mpDevice, kMultiReprojectTemporalSamplesFile, "multiReprojectTemporalSamples", "", mpScene, defines, globalTypeConformances);
    mpMultiReprojectTemporalSamplesPass->prepareProgram(mpDevice, defines);

    if (!mpComputeCellOffsetsPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSortReprojectedReservoirsFile).csEntry("computeCellOffsets");
        mpComputeCellOffsetsPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpSortCellDataPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kSortReprojectedReservoirsFile).csEntry("sortCellData");
        mpSortCellDataPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpMultiComputeCellOffsetsPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSortReprojectedReservoirsFile).csEntry("computeCellOffsets");
        mpMultiComputeCellOffsetsPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpMultiSortCellDataPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kMultiSortReprojectedReservoirsFile).csEntry("sortCellData");
        mpMultiSortCellDataPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpVisualizeForwardReprojectionPass)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kVisualizeForwardReprojectionFile).csEntry("main");
        mpVisualizeForwardReprojectionPass = ComputePass::create(mpDevice, desc, defines, false);
    }

    if (!mpGatherTemporalResamplingPass)
        mpGatherTemporalResamplingPass = std::make_unique<TracePass>(mpDevice, kGatherTemporalResamplingFile, "gatherTemporalResampling", "", mpScene, defines, globalTypeConformances);
    mpGatherTemporalResamplingPass->prepareProgram(mpDevice, defines);

    if (!mpScatterTemporalResamplingPass)
        mpScatterTemporalResamplingPass = std::make_unique<TracePass>(mpDevice, kScatterTemporalResamplingFile, "scatterTemporalResampling", "", mpScene, defines, globalTypeConformances);
    mpScatterTemporalResamplingPass->prepareProgram(mpDevice, defines);

    if (!mpMultiScatterTemporalResamplingPass)
        mpMultiScatterTemporalResamplingPass = std::make_unique<TracePass>(mpDevice, kMultiScatterTemporalResamplingFile, "multiScatterTemporalResampling", "", mpScene, defines, globalTypeConformances);
    mpMultiScatterTemporalResamplingPass->prepareProgram(mpDevice, defines);

    if (!mpScatterBackupTemporalResamplingPass)
        mpScatterBackupTemporalResamplingPass = std::make_unique<TracePass>(mpDevice, kScatterBackupTemporalResamplingFile, "scatterBackupTemporalResampling", "", mpScene, defines, globalTypeConformances);
    mpScatterBackupTemporalResamplingPass->prepareProgram(mpDevice, defines);

    if (!mpSpatialResamplingPass)
        mpSpatialResamplingPass = std::make_unique<TracePass>(mpDevice, kSpatialResamplingFile, "spatialResampling", "", mpScene, defines, globalTypeConformances);
    mpSpatialResamplingPass->prepareProgram(mpDevice, defines);

    if (!mpReflectTypes)
    {
        ProgramDesc desc = baseDesc;
        desc.addShaderLibrary(kReflectTypesFile).csEntry("main");
        mpReflectTypes = ComputePass::create(mpDevice, desc, defines, false);
    }

    auto preparePass = [&](ref<ComputePass> pass)
    {
        // Note that we must use set instead of add defines to replace any stale state.
        pass->getProgram()->setDefines(defines);

        // Recreate program vars. This may trigger recompilation if needed.
        // Note that program versions are cached, so switching to a previously used specialization is faster.
        pass->setVars(nullptr);
    };
    preparePass(mpCollectTemporalSamplesPass);
    preparePass(mpComputeCellOffsetsPass);
    preparePass(mpSortCellDataPass);
    preparePass(mpMultiComputeCellOffsetsPass);
    preparePass(mpMultiSortCellDataPass);
    preparePass(mpVisualizeForwardReprojectionPass);
    preparePass(mpResolveReSTIRPass);
    preparePass(mpReflectTypes);

    mVarsChanged = true;
    mRecompile = false;
}

ref<Texture> ReservoirSplatting::createNeighborOffsetTexture(const uint32_t sampleCount)
{
    std::unique_ptr<int8_t[]> offsets(new int8_t[sampleCount * 2]);
    const int R = 254;
    const float phi2 = 1.f / 1.3247179572447f;
    float u = 0.5f;
    float v = 0.5f;
    for (uint32_t index = 0; index < sampleCount * 2;)
    {
        u += phi2;
        v += phi2 * phi2;
        if (u >= 1.f)
            u -= 1.f;
        if (v >= 1.f)
            v -= 1.f;

        float rSq = (u - 0.5f) * (u - 0.5f) + (v - 0.5f) * (v - 0.5f);
        if (rSq > 0.25f)
            continue;

        offsets[index++] = int8_t((u - 0.5f) * R);
        offsets[index++] = int8_t((v - 0.5f) * R);
    }

    return mpDevice->createTexture1D(sampleCount, ResourceFormat::RG8Snorm, 1, 1, offsets.get());
}

void ReservoirSplatting::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_ASSERT(mpScene);

    // Compute allocation requirements for paths and output samples.
    // Note that the sample buffers are padded to whole tiles, while the max path count depends on actual frame dimension.
    // If we don't have a fixed sample count, assume the worst case.
    uint32_t spp = mStaticParams.samplesPerPixel;
    uint32_t tileCount = mParams.screenTiles.x * mParams.screenTiles.y;
    const uint32_t sampleCount = tileCount * kScreenTileDim.x * kScreenTileDim.y * spp;
    const uint32_t screenPixelCount = mParams.frameDim.x * mParams.frameDim.y;
    const uint32_t pathCount = screenPixelCount * spp;
    auto var = mpReflectTypes->getRootVar();

    // Allocate buffers used for ReSTIR.
    if (!mpFloatingCoordinates || (mpFloatingCoordinates->getWidth() != mParams.frameDim.x) || (mpFloatingCoordinates->getHeight() != mParams.frameDim.y) ||
        mVarsChanged)
    {
        mpFloatingCoordinates = mpDevice->createTexture2D(
            mParams.frameDim.x,
            mParams.frameDim.y,
            ResourceFormat::RG32Float, // (x, y) coordinates in floating-point.
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    }

    if (!mpShiftedPaths || (mpShiftedPaths->getElementCount() != 8 * screenPixelCount) || mVarsChanged)
    {
        mpShiftedPaths = mpDevice->createStructuredBuffer(
            var["shiftedPaths"],
            8 * screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }

    if (!mpCurrReservoirs || (mpCurrReservoirs->getElementCount() != screenPixelCount) ||
        !mpIntermediateReservoirs || (mpIntermediateReservoirs->getElementCount() != screenPixelCount) ||
        !mpPrevReservoirs || (mpPrevReservoirs->getElementCount() != screenPixelCount) ||
        mVarsChanged)
    {
        mpPrevReservoirs = mpDevice->createStructuredBuffer(
            var["pathReservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpIntermediateReservoirs = mpDevice->createStructuredBuffer(
            var["pathReservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpCurrReservoirs = mpDevice->createStructuredBuffer(
            var["pathReservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpTempReservoirs = mpDevice->createStructuredBuffer(
            var["pathReservoirs"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }

    if (!mpGlobalCounters || !mpCellCounters || !mpTotalCellCounters || !mpStagingTotalCellCounters ||!mpReadbackFence ||
        !mpReservoirIndices || (mpReservoirIndices->getElementCount() != screenPixelCount) ||
        !mpScatteredReservoirs || (mpScatteredReservoirs->getElementCount() != screenPixelCount) ||
        !mpCellOffsets || (mpCellOffsets->getElementCount() != screenPixelCount) ||
        !mpSortedReservoirs || (mpSortedReservoirs->getElementCount() != screenPixelCount))
    {
        mpGlobalCounters = mpDevice->createBuffer(sizeof(uint32_t) * 2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpCellCounters = mpDevice->createBuffer(sizeof(uint32_t) * screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpTotalCellCounters = mpDevice->createBuffer(sizeof(float) * screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess);
        mpStagingTotalCellCounters = mpDevice->createBuffer(sizeof(float) * screenPixelCount, ResourceBindFlags::None, MemoryType::ReadBack);
        mpReadbackFence = mpDevice->createFence();

        mpReservoirIndices = mpDevice->createStructuredBuffer(
            var["buffer2D"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpScatteredReservoirs = mpDevice->createStructuredBuffer(
            var["buffer2D"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpCellOffsets = mpDevice->createStructuredBuffer(
            var["buffer1D"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpSortedReservoirs = mpDevice->createStructuredBuffer(
            var["buffer2D"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }

    if (mMultiGlobalCounters.empty() || mMultiCellCounters.empty() ||
        mMultiGlobalCounters.size() != mReSTIRParams.numTimePartitions || mMultiCellCounters.size() != mReSTIRParams.numTimePartitions)
    {
        mMultiGlobalCounters.clear();
        mMultiCellCounters.clear();
        mMultiReservoirIndices.clear();
        mMultiScatteredReservoirs.clear();
        mMultiCellOffsets.clear();
        mMultiSortedReservoirs.clear();

        for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
        {
            mMultiGlobalCounters.push_back(
                mpDevice->createBuffer(sizeof(uint32_t) * 2, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
            );

            mMultiCellCounters.push_back(
                mpDevice->createBuffer(sizeof(uint32_t) * screenPixelCount, ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess)
            );

            mMultiReservoirIndices.push_back(
                mpDevice->createStructuredBuffer(
                    var["buffer2D"],
                    screenPixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                )
            );

            mMultiScatteredReservoirs.push_back(
                mpDevice->createStructuredBuffer(
                    var["buffer2D"],
                    screenPixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                )
            );

            mMultiCellOffsets.push_back(
                mpDevice->createStructuredBuffer(
                    var["buffer1D"],
                    screenPixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                )
            );

            mMultiSortedReservoirs.push_back(
                mpDevice->createStructuredBuffer(
                    var["buffer2D"],
                    screenPixelCount,
                    ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
                    MemoryType::DeviceLocal,
                    nullptr,
                    false
                )
            );
        }
    }

    if (!mpCurrReconnectionData || (mpCurrReconnectionData->getElementCount() != screenPixelCount) ||
        !mpIntermediateReconnectionData || (mpIntermediateReconnectionData->getElementCount() != screenPixelCount) ||
        !mpPrevReconnectionData || (mpPrevReconnectionData->getElementCount() != screenPixelCount) ||
        mVarsChanged)
    {
        mpPrevReconnectionData = mpDevice->createStructuredBuffer(
            var["reconnectionData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpIntermediateReconnectionData = mpDevice->createStructuredBuffer(
            var["reconnectionData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpCurrReconnectionData = mpDevice->createStructuredBuffer(
            var["reconnectionData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );

        mpTempReconnectionData = mpDevice->createStructuredBuffer(
            var["reconnectionData"],
            screenPixelCount,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    }

    if (!mpNeighborOffsets)
    {
        mpNeighborOffsets = createNeighborOffsetTexture(kSpatialNeighborSamples);
    }
}

void ReservoirSplatting::preparePathTracer(const RenderData& renderData)
{
    // Create path tracer parameter block if needed.
    if (!mpPathTracerBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("pathTracer");
        mpPathTracerBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpPathTracerBlock);
    }

    // Bind resources.
    auto var = mpPathTracerBlock->getRootVar();
    bindShaderData(var, renderData);
}

void ReservoirSplatting::prepareCameraManager(const RenderData& renderData, bool shouldReset)
{
    // Create the camera manager parameter block if needed.
    if (!mpCameraManagerBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("cameraManager");
        mpCameraManagerBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpCameraManagerBlock);
    }

    // Bind whatever we need
    auto var = mpCameraManagerBlock->getRootVar();
    var["artificialFrameTime"] = mCameraParams.artificialFrameTime;

    if (mParams.frameCount == 0 || shouldReset)
    {
        for (int i = 0; i < mCameraParams.currNumKeyframes; i++)
        {
            var["cameraPositions"][i] = mpScene->getCamera()->getPosition();
            var["cameraTargets"][i] = mpScene->getCamera()->getTarget();
        }
    }
    else
    {
        int currIndex = mParams.frameCount % mCameraParams.currNumKeyframes;
        var["cameraPositions"][currIndex] = mpScene->getCamera()->getPosition();
        var["cameraTargets"][currIndex] = mpScene->getCamera()->getTarget();
    }
}

void ReservoirSplatting::prepareGatherData(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Create the gather data parameter block if needed.
    if (!mpGatherDataBlock || mVarsChanged)
    {
        auto reflector = mpReflectTypes->getProgram()->getReflector()->getParameterBlock("gatherData");
        mpGatherDataBlock = ParameterBlock::create(mpDevice, reflector);
        FALCOR_ASSERT(mpGatherDataBlock);
    }

    // Bind whatever we need.
    auto var = mpGatherDataBlock->getRootVar();
    var["gatherOption"] = static_cast<uint32_t>(mReSTIRParams.temporalGathering);
    var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["floatingCoords"] = mpFloatingCoordinates;
}

void ReservoirSplatting::resetLighting()
{
    // Retain the options for the emissive sampler.
    if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
    {
        mLightBVHOptions = lightBVHSampler->getOptions();
    }

    mpEmissiveSampler = nullptr;
    mpEnvMapSampler = nullptr;
    mRecompile = true;
}

void ReservoirSplatting::prepareMaterials(RenderContext* pRenderContext)
{
    // This functions checks for scene changes that require shader recompilation.
    // Whenever materials or geometry is added/removed to the scene, we reset the shader programs to trigger
    // recompilation with the correct defines, type conformances, shader modules, and binding table.

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RecompileNeeded) ||
        is_set(mpScene->getUpdates(), Scene::UpdateFlags::GeometryChanged))
    {
        resetPrograms();
    }
}

bool ReservoirSplatting::prepareLighting(RenderContext* pRenderContext)
{
    bool lightingChanged = false;

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::RenderSettingsChanged))
    {
        lightingChanged = true;
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::SDFGridConfigChanged))
    {
        mRecompile = true;
    }

    if (is_set(mpScene->getUpdates(), Scene::UpdateFlags::EnvMapChanged))
    {
        mpEnvMapSampler = nullptr;
        lightingChanged = true;
        mRecompile = true;
    }

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEnvMapSampler)
        {
            mpEnvMapSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    // Request the light collection if emissive lights are enabled.
    if (mpScene->getRenderSettings().useEmissiveLights)
    {
        mpScene->getLightCollection(pRenderContext);
    }

    if (mpScene->useEmissiveLights())
    {
        if (!mpEmissiveSampler)
        {
            const auto& pLights = mpScene->getLightCollection(pRenderContext);
            FALCOR_ASSERT(pLights && pLights->getActiveLightCount(pRenderContext) > 0);
            FALCOR_ASSERT(!mpEmissiveSampler);

            switch (mStaticParams.emissiveSampler)
            {
            case EmissiveLightSamplerType::Uniform:
                mpEmissiveSampler = std::make_unique<EmissiveUniformSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            case EmissiveLightSamplerType::LightBVH:
                mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext), mLightBVHOptions);
                break;
            case EmissiveLightSamplerType::Power:
                mpEmissiveSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, mpScene->getILightCollection(pRenderContext));
                break;
            default:
                FALCOR_THROW("Unknown emissive light sampler type");
            }
            lightingChanged = true;
            mRecompile = true;
        }
    }
    else
    {
        if (mpEmissiveSampler)
        {
            // Retain the options for the emissive sampler.
            if (auto lightBVHSampler = dynamic_cast<LightBVHSampler*>(mpEmissiveSampler.get()))
            {
                mLightBVHOptions = lightBVHSampler->getOptions();
            }

            mpEmissiveSampler = nullptr;
            lightingChanged = true;
            mRecompile = true;
        }
    }

    if (mpEmissiveSampler)
    {
        lightingChanged |= mpEmissiveSampler->update(pRenderContext, mpScene->getILightCollection(pRenderContext));
        // auto defines = mpEmissiveSampler->getDefines();
        // if (mpInitialCandidatesPass && mpInitialCandidatesPass->addDefines(defines)) mRecompile = true;
    }

    return lightingChanged;
}

void ReservoirSplatting::bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling) const
{
    // Bind static resources that don't change per frame.
    if (mVarsChanged)
    {
        if (useLightSampling && mpEnvMapSampler) mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
    }

    // Bind runtime data.
    var["params"].setBlob(mParams);

    if (useLightSampling && mpEmissiveSampler)
    {
        // TODO: Do we have to bind this every frame?
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    }
}

bool ReservoirSplatting::beginFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutputColor = renderData.getTexture(kOutputColor);
    FALCOR_ASSERT(pOutputColor);

    // Set output frame dimension.
    setFrameDim(uint2(pOutputColor->getWidth(), pOutputColor->getHeight()));

    // Validate all I/O sizes match the expected size.
    // If not, we'll disable the path tracer to give the user a chance to fix the configuration before re-enabling it.
    bool resolutionMismatch = false;
    auto validateChannels = [&](const auto& channels) {
        for (const auto& channel : channels)
        {
            auto pTexture = renderData.getTexture(channel.name);
            if (pTexture && (pTexture->getWidth() != mParams.frameDim.x || pTexture->getHeight() != mParams.frameDim.y)) resolutionMismatch = true;
        }
    };
    // validateChannels(kInputChannels);
    validateChannels(kOutputChannels);

    if (mEnabled && resolutionMismatch)
    {
        logError("ReservoirSplatting I/O sizes don't match. The pass will be disabled.");
        mEnabled = false;
    }

    if (mpScene == nullptr || !mEnabled)
    {
        pRenderContext->clearUAV(pOutputColor->getUAV().get(), float4(0.f));

        // Set refresh flag if changes that affect the output have occured.
        // This is needed to ensure other passes get notified when the path tracer is enabled/disabled.
        if (mOptionsChanged)
        {
            auto& dict = renderData.getDictionary();
            auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
            if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
            dict[Falcor::kRenderPassRefreshFlags] = flags;
        }

        return false;
    }

    // Update materials.
    prepareMaterials(pRenderContext);

    // Update the env map and emissive sampler to the current frame.
    bool lightingChanged = prepareLighting(pRenderContext);

    // Update refresh flag if changes that affect the output have occured.
    auto& dict = renderData.getDictionary();
    if (mOptionsChanged || lightingChanged)
    {
        auto flags = dict.getValue(kRenderPassRefreshFlags, Falcor::RenderPassRefreshFlags::None);
        if (mOptionsChanged) flags |= Falcor::RenderPassRefreshFlags::RenderOptionsChanged;
        if (lightingChanged) flags |= Falcor::RenderPassRefreshFlags::LightingChanged;
        dict[Falcor::kRenderPassRefreshFlags] = flags;
        mOptionsChanged = false;
    }

    // Check if GBuffer has adjusted shading normals enabled.
    bool gbufferAdjustShadingNormals = dict.getValue(Falcor::kRenderPassGBufferAdjustShadingNormals, false);
    if (gbufferAdjustShadingNormals != mGBufferAdjustShadingNormals)
    {
        mGBufferAdjustShadingNormals = gbufferAdjustShadingNormals;
        mRecompile = true;
    }

    // Enable pixel stats if rayCount or pathLength outputs are connected.
    if (renderData[kOutputRayCount] != nullptr || renderData[kOutputPathLength] != nullptr)
    {
        mpPixelStats->setEnabled(true);
    }

    mpPixelStats->beginFrame(pRenderContext, mParams.frameDim);
    mpPixelDebug->beginFrame(pRenderContext, mParams.frameDim);

    // Update the random seed.
    mParams.seed = mParams.useFixedSeed ? mParams.fixedSeed : mParams.seed + 1;

    // We should make sure that the required number of keyframes is up to date,
    // which requires recompilation to re-create the static arrays.
    mCameraParams.currNumKeyframes = 1 + ceil(mpScene->getCamera()->getShutterSpeed() / mCameraParams.artificialFrameTime);
    // If we're doing a reconnection shift, store an extra sample (so that the previous frame's trajectory is encompassed as well).
    // But to avoid recompiling between shifts, just always have it available...
    mCameraParams.currNumKeyframes++;

    if (mCameraParams.prevNumKeyframes != mCameraParams.currNumKeyframes)
    {
        mRecompile = true;
    }

    mCameraParams.prevNumKeyframes = mCameraParams.currNumKeyframes;

    return true;
}

void ReservoirSplatting::endFrame(RenderContext* pRenderContext, const RenderData& renderData)
{
    mpPixelStats->endFrame(pRenderContext);
    mpPixelDebug->endFrame(pRenderContext);

    auto copyTexture = [pRenderContext](Texture* pDst, const Texture* pSrc)
    {
        if (pDst && pSrc)
        {
            FALCOR_ASSERT(pDst && pSrc);
            FALCOR_ASSERT(pDst->getFormat() == pSrc->getFormat());
            FALCOR_ASSERT(pDst->getWidth() == pSrc->getWidth() && pDst->getHeight() == pSrc->getHeight());
            pRenderContext->copyResource(pDst, pSrc);
        }
        else if (pDst)
        {
            pRenderContext->clearUAV(pDst->getUAV().get(), uint4(0, 0, 0, 0));
        }
    };

    // Copy pixel stats to outputs if available.
    copyTexture(renderData.getTexture(kOutputRayCount).get(), mpPixelStats->getRayCountTexture(pRenderContext).get());
    copyTexture(renderData.getTexture(kOutputPathLength).get(), mpPixelStats->getPathLengthTexture().get());

    if (!mpScene->isFrozen())
    {
        mParams.frameCount++;

        // Swap any current buffers that are now the previous buffers of the next frame.
        std::swap(mpCurrReservoirs, mpPrevReservoirs);
        std::swap(mpCurrReconnectionData, mpPrevReconnectionData);
    }
}

void ReservoirSplatting::tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass, uint z)
{
    FALCOR_PROFILE(pRenderContext, tracePass.name);

    FALCOR_ASSERT(tracePass.pProgram != nullptr && tracePass.pBindingTable != nullptr && tracePass.pVars != nullptr);

    // Bind global resources.
    auto var = tracePass.pVars->getRootVar();
    mpScene->bindShaderDataForRaytracing(pRenderContext, var["gScene"], 2);

    if (mVarsChanged) mpSampleGenerator->bindShaderData(var);

    mpPixelStats->prepareProgram(tracePass.pProgram, var);
    mpPixelDebug->prepareProgram(tracePass.pProgram, var);

    // Bind the path tracer and camera manager.
    var["gPathTracer"] = mpPathTracerBlock;
    var["gCameraManager"] = mpCameraManagerBlock;

    // Full screen dispatch.
    mpScene->raytrace(pRenderContext, tracePass.pProgram.get(), tracePass.pVars, uint3(mParams.frameDim, z));
}

void ReservoirSplatting::initialCandidateGeneration(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "initialCandidates");

    auto rootVar = mpInitialCandidatesPass->getRootVar();
    auto var = rootVar["CB"]["gInitialCandidates"];//pVars->getRootVar()["CB"]["gInitialCandidates"];
    mpScene->bindShaderData(rootVar["gScene"]);
    mpScene->bindShaderDataForRaytracing(pRenderContext, rootVar["gScene"]);
    var["params"].setBlob(mParams);
    var["currReservoirs"] = mpCurrReservoirs;
    var["currReconnectionData"] = mpCurrReconnectionData;
    rootVar["gPathTracer"] = mpPathTracerBlock;
    rootVar["gCameraManager"] = mpCameraManagerBlock;
    mpPixelStats->prepareProgram(mpInitialCandidatesPass->getProgram(), rootVar);
    mpPixelDebug->prepareProgram(mpInitialCandidatesPass->getProgram(), rootVar);
    mpInitialCandidatesPass->execute(pRenderContext, {mParams.frameDim.x, mParams.frameDim.y, 1});
    //tracePass(pRenderContext, renderData, *mpInitialCandidatesPass);
}

void ReservoirSplatting::robustReuseOptimization(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpRobustReuseOptimizationPass->pVars->getRootVar()["CB"]["gRobustReuseOptimization"];
    var["params"].setBlob(mParams);
    var["prevReservoirs"] = mpPrevReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;
    var["shiftedPaths"] = mpShiftedPaths;

    tracePass(pRenderContext, renderData, *mpRobustReuseOptimizationPass);
}

void ReservoirSplatting::collectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "collectTemporalSamples");

    auto vars = mpCollectTemporalSamplesPass->getRootVar();
    mpPixelDebug->prepareProgram(mpCollectTemporalSamplesPass->getProgram(), vars);

    vars["gCameraManager"] = mpCameraManagerBlock;

    auto var = vars["CB"]["gCollectTemporalSamples"];
    var["params"].setBlob(mParams);
    var["gatherOption"] = static_cast<uint32_t>(mReSTIRParams.temporalGathering);
    var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["prevReservoirs"] = mpPrevReservoirs;
    var["intermediateReservoirs"] = mpIntermediateReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;
    var["intermediateReconnectionData"] = mpIntermediateReconnectionData;
    var["floatingCoords"] = mpFloatingCoordinates;
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsTemporally;
    var["shiftedPaths"] = mpShiftedPaths;

    mpCollectTemporalSamplesPass->execute(pRenderContext, {mParams.frameDim, 1u});
}

void ReservoirSplatting::reprojectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData)
{
    // Make sure the global counters are reset.
    pRenderContext->clearUAV(mpGlobalCounters->getUAV().get(), uint4(0));
    pRenderContext->clearUAV(mpCellCounters->getUAV().get(), uint4(0));

    auto var = mpReprojectTemporalSamplesPass->pVars->getRootVar()["CB"]["gReprojectTemporalSamples"];
    var["params"].setBlob(mParams);
    var["globalCounters"] = mpGlobalCounters;
    var["cellCounters"] = mpCellCounters;
    var["reservoirIndices"] = mpReservoirIndices;
    var["scatteredReservoirs"] = mpScatteredReservoirs;
    var["prevReservoirs"] = mpPrevReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;

    tracePass(pRenderContext, renderData, *mpReprojectTemporalSamplesPass);
}

void ReservoirSplatting::multiReprojectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpMultiReprojectTemporalSamplesPass->pVars->getRootVar()["CB"]["gMultiReprojectTemporalSamples"];
    var["params"].setBlob(mParams);

    for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
    {
        pRenderContext->clearUAV(mMultiGlobalCounters[i]->getUAV().get(), uint4(0));
        pRenderContext->clearUAV(mMultiCellCounters[i]->getUAV().get(), uint4(0));

        var["globalCounters"][i] = mMultiGlobalCounters[i];
        var["cellCounters"][i] = mMultiCellCounters[i];
        var["reservoirIndices"][i] = mMultiReservoirIndices[i];
        var["scatteredReservoirs"][i] = mMultiScatteredReservoirs[i];
    }

    var["prevReservoirs"] = mpPrevReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;

    tracePass(pRenderContext, renderData, *mpMultiReprojectTemporalSamplesPass);
}

void ReservoirSplatting::sortReprojectedReservoirs(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint32_t cellCount = mParams.frameDim.x * mParams.frameDim.y;

    {
        FALCOR_PROFILE(pRenderContext, "computeCellOffsets");

        auto vars = mpComputeCellOffsetsPass->getRootVar();
        mpPixelDebug->prepareProgram(mpComputeCellOffsetsPass->getProgram(), vars);

        auto var = vars["CB"]["gSortReprojectedReservoirs"];
        var["params"].setBlob(mParams);
        var["cellCount"] = cellCount;
        var["globalCounters"] = mpGlobalCounters;
        var["cellCounters"] = mpCellCounters;
        var["reservoirIndices"] = mpReservoirIndices;
        var["scatteredReservoirs"] = mpScatteredReservoirs;
        var["cellOffsets"] = mpCellOffsets;
        var["sortedReservoirs"] = mpSortedReservoirs;

        mpComputeCellOffsetsPass->execute(pRenderContext, {mParams.frameDim, 1u});
    }

    {
        FALCOR_PROFILE(pRenderContext, "sortCellData");

        auto vars = mpSortCellDataPass->getRootVar();
        mpPixelDebug->prepareProgram(mpSortCellDataPass->getProgram(), vars);

        auto var = vars["CB"]["gSortReprojectedReservoirs"];
        var["params"].setBlob(mParams);
        var["cellCount"] = cellCount;
        var["globalCounters"] = mpGlobalCounters;
        var["cellCounters"] = mpCellCounters;
        var["reservoirIndices"] = mpReservoirIndices;
        var["scatteredReservoirs"] = mpScatteredReservoirs;
        var["cellOffsets"] = mpCellOffsets;
        var["sortedReservoirs"] = mpSortedReservoirs;

        mpSortCellDataPass->execute(pRenderContext, {cellCount, 1u, 1u});
    }
}

void ReservoirSplatting::multiSortReprojectedReservoirs(RenderContext* pRenderContext, const RenderData& renderData)
{
    uint32_t cellCount = mParams.frameDim.x * mParams.frameDim.y;

    {
        FALCOR_PROFILE(pRenderContext, "multiComputeCellOffsets");

        auto vars = mpMultiComputeCellOffsetsPass->getRootVar();
        mpPixelDebug->prepareProgram(mpMultiComputeCellOffsetsPass->getProgram(), vars);

        auto var = vars["CB"]["gMultiSortReprojectedReservoirs"];
        var["params"].setBlob(mParams);
        var["cellCount"] = cellCount;

        for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
        {
            var["globalCounters"][i] = mMultiGlobalCounters[i];
            var["cellCounters"][i] = mMultiCellCounters[i];
            var["reservoirIndices"][i] = mMultiReservoirIndices[i];
            var["scatteredReservoirs"][i] = mMultiScatteredReservoirs[i];
            var["cellOffsets"][i] = mMultiCellOffsets[i];
            var["sortedReservoirs"][i] = mMultiSortedReservoirs[i];
        }

        mpMultiComputeCellOffsetsPass->execute(pRenderContext, {mParams.frameDim, 1u});
    }

    {
        FALCOR_PROFILE(pRenderContext, "multiSortCellData");

        auto vars = mpMultiSortCellDataPass->getRootVar();
        mpPixelDebug->prepareProgram(mpMultiSortCellDataPass->getProgram(), vars);

        auto var = vars["CB"]["gMultiSortReprojectedReservoirs"];
        var["params"].setBlob(mParams);
        var["cellCount"] = cellCount;

        for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
        {
            var["globalCounters"][i] = mMultiGlobalCounters[i];
            var["cellCounters"][i] = mMultiCellCounters[i];
            var["reservoirIndices"][i] = mMultiReservoirIndices[i];
            var["scatteredReservoirs"][i] = mMultiScatteredReservoirs[i];
            var["cellOffsets"][i] = mMultiCellOffsets[i];
            var["sortedReservoirs"][i] = mMultiSortedReservoirs[i];
        }

        mpMultiSortCellDataPass->execute(pRenderContext, {cellCount, 1u, 1u});
    }
}

void ReservoirSplatting::visualizeForwardReprojection(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "visualizeForwardReprojection");

    pRenderContext->clearUAV(mpTotalCellCounters->getUAV().get(), uint4(0));

    auto vars = mpVisualizeForwardReprojectionPass->getRootVar();
    mpPixelDebug->prepareProgram(mpVisualizeForwardReprojectionPass->getProgram(), vars);

    auto var = vars["CB"]["gVisualizeForwardReprojection"];
    var["params"].setBlob(mParams);
    var["isMultiScatter"] = (mReSTIRParams.temporalReuseOption == TemporalReuse::MultiScatter);
    var["cellCounters"] = mpCellCounters;
    for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
    {
        var["multiCellCounters"][i] = mMultiCellCounters[i];
    }

    var["totalScattered"] = mpTotalCellCounters;

    mpVisualizeForwardReprojectionPass->execute(pRenderContext, {mParams.frameDim, 1u});
}

void ReservoirSplatting::gatherTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto vars = mpGatherTemporalResamplingPass->pVars->getRootVar();
    vars["gGatherData"] = mpGatherDataBlock;

    auto var = vars["CB"]["gGatherTemporalResampling"];
    var["params"].setBlob(mParams);
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsTemporally;
    var["prevReservoirs"] = mpIntermediateReservoirs;
    var["currReservoirs"] = mpCurrReservoirs;
    var["prevReconnectionData"] = mpIntermediateReconnectionData;
    var["currReconnectionData"] = mpCurrReconnectionData;
    var["vbuffer"] = renderData.getTexture(kInputVBuffer);

    tracePass(pRenderContext, renderData, *mpGatherTemporalResamplingPass);
}

void ReservoirSplatting::scatterTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpScatterTemporalResamplingPass->pVars->getRootVar()["CB"]["gScatterTemporalResampling"];
    var["params"].setBlob(mParams);
    var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsTemporally;
    var["cellCounters"] = mpCellCounters;
    var["cellOffsets"] = mpCellOffsets;
    var["sortedReservoirs"] = mpSortedReservoirs;
    var["prevReservoirs"] = mpPrevReservoirs;
    var["currReservoirs"] = mpCurrReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;
    var["currReconnectionData"] = mpCurrReconnectionData;

    tracePass(pRenderContext, renderData, *mpScatterTemporalResamplingPass);
}

void ReservoirSplatting::multiScatterTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpMultiScatterTemporalResamplingPass->pVars->getRootVar()["CB"]["gMultiScatterTemporalResampling"];
    var["params"].setBlob(mParams);
    var["motionVectors"] = renderData.getTexture(kInputMotionVectors);
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsTemporally;

    for (uint32_t i = 0; i < mReSTIRParams.numTimePartitions; i++)
    {
        var["cellCounters"][i] = mMultiCellCounters[i];
        var["cellOffsets"][i] = mMultiCellOffsets[i];
        var["sortedReservoirs"][i] = mMultiSortedReservoirs[i];
    }

    var["prevReservoirs"] = mpPrevReservoirs;
    var["currReservoirs"] = mpCurrReservoirs;
    var["prevReconnectionData"] = mpPrevReconnectionData;
    var["currReconnectionData"] = mpCurrReconnectionData;

    tracePass(pRenderContext, renderData, *mpMultiScatterTemporalResamplingPass);
}

void ReservoirSplatting::scatterBackupTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto vars = mpScatterBackupTemporalResamplingPass->pVars->getRootVar();
    vars["gGatherData"] = mpGatherDataBlock;

    auto var = vars["CB"]["gScatterBackupTemporalResampling"];
    var["params"].setBlob(mParams);
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsTemporally;
    var["misOption"] = static_cast<uint32_t>(mReSTIRParams.scatterBackupMISOption);
    var["cellCounters"] = mpCellCounters;
    var["cellOffsets"] = mpCellOffsets;
    var["sortedReservoirs"] = mpSortedReservoirs;
    var["backupReservoirs"] = mpIntermediateReservoirs;
    var["prevReservoirs"] = mpPrevReservoirs;
    var["currReservoirs"] = mpCurrReservoirs;
    var["backupReconnectionData"] = mpIntermediateReconnectionData;
    var["prevReconnectionData"] = mpPrevReconnectionData;
    var["currReconnectionData"] = mpCurrReconnectionData;

    tracePass(pRenderContext, renderData, *mpScatterBackupTemporalResamplingPass);
}

void ReservoirSplatting::spatialResampling(RenderContext* pRenderContext, const RenderData& renderData)
{
    auto var = mpSpatialResamplingPass->pVars->getRootVar()["CB"]["gSpatialResampling"];
    var["params"].setBlob(mParams);
    var["useConfidenceWeights"] = mReSTIRParams.useConfidenceWeightsSpatially;
    var["neighborCount"] = mReSTIRParams.neighborCount;
    var["neighborOffsets"] = mpNeighborOffsets;
    var["gatherRadius"] = mReSTIRParams.spatialGatherRadius;

    ref<Buffer>& pSwapReservoirs = mpScene->isFrozen() ? mpTempReservoirs : mpPrevReservoirs;
    ref<Buffer>& pSwapReconnectionData = mpScene->isFrozen() ? mpTempReconnectionData : mpPrevReconnectionData;

    for (uint32_t i = 0; i < mReSTIRParams.numSpatialIterations; i++)
    {
        std::swap(mpCurrReconnectionData, pSwapReconnectionData);
        std::swap(mpCurrReservoirs, pSwapReservoirs);

        var["iteration"] = i;
        var["prevReservoirs"] = pSwapReservoirs;
        var["currReservoirs"] = mpCurrReservoirs;
        var["prevReconnectionData"] = pSwapReconnectionData;
        var["currReconnectionData"] = mpCurrReconnectionData;
        var["vbuffer"] = renderData.getTexture(kInputVBuffer);

        tracePass(pRenderContext, renderData, *mpSpatialResamplingPass);
    }
}

void ReservoirSplatting::resolveReSTIR(RenderContext* pRenderContext, const RenderData& renderData)
{
    FALCOR_PROFILE(pRenderContext, "resolveReSTIR");

    auto varsReSTIR = mpResolveReSTIRPass->getRootVar();
    mpPixelDebug->prepareProgram(mpResolveReSTIRPass->getProgram(), varsReSTIR);

    // Bind resources.
    auto var = varsReSTIR["CB"]["gResolveReSTIR"];
    var["params"].setBlob(mParams);
    var["currReservoirs"] = mpCurrReservoirs;
    var["outputColor"] = renderData.getTexture(kOutputColor);

    // Launch one thread per pixel.
    mpResolveReSTIRPass->execute(pRenderContext, {mParams.frameDim, 1u});
}

DefineList ReservoirSplatting::StaticParams::getDefines(const ReservoirSplatting& owner) const
{
    DefineList defines;

    // Path tracer configuration.
    defines.add("SAMPLES_PER_PIXEL", std::to_string(samplesPerPixel)); // 0 indicates a variable sample count
    defines.add("NUM_RENDER_PASSES", std::to_string(kNumPassesWithRNG));
    defines.add("MAX_SURFACE_BOUNCES", std::to_string(maxSurfaceBounces));
    defines.add("MAX_DIFFUSE_BOUNCES", std::to_string(maxDiffuseBounces));
    defines.add("MAX_SPECULAR_BOUNCES", std::to_string(maxSpecularBounces));
    defines.add("MAX_TRANSMISSON_BOUNCES", std::to_string(maxTransmissionBounces));
    defines.add("ADJUST_SHADING_NORMALS", adjustShadingNormals ? "1" : "0");
    defines.add("USE_RUSSIAN_ROULETTE", useRussianRoulette ? "1" : "0");
    defines.add("USE_ALPHA_TEST", useAlphaTest ? "1" : "0");
    defines.add("USE_LIGHTS_IN_DIELECTRIC_VOLUMES", useLightsInDielectricVolumes ? "1" : "0");
    defines.add("DISABLE_CAUSTICS", disableCaustics ? "1" : "0");
    defines.add("PRIMARY_LOD_MODE", std::to_string((uint32_t)primaryLodMode));
    defines.add("COLOR_FORMAT", std::to_string((uint32_t)colorFormat));
    defines.add("MIS_HEURISTIC", std::to_string((uint32_t)misHeuristic));
    defines.add("MIS_POWER_EXPONENT", std::to_string(misPowerExponent));
    defines.add("SPATIAL_SAMPLE_COUNT", std::to_string(kSpatialNeighborSamples));

    // Sampling utilities configuration.
    FALCOR_ASSERT(owner.mpSampleGenerator);
    defines.add(owner.mpSampleGenerator->getDefines());

    if (owner.mpEmissiveSampler) defines.add(owner.mpEmissiveSampler->getDefines());

    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(maxNestedMaterials));

    defines.add("GBUFFER_ADJUST_SHADING_NORMALS", owner.mGBufferAdjustShadingNormals ? "1" : "0");

    // Scene-specific configuration.
    const auto& scene = owner.mpScene;
    if (scene) defines.add(scene->getSceneDefines());
    defines.add("USE_ENV_LIGHT", scene && scene->useEnvLight() ? "1" : "0");
    defines.add("USE_ANALYTIC_LIGHTS", scene && scene->useAnalyticLights() ? "1" : "0");
    defines.add("USE_EMISSIVE_LIGHTS", scene && scene->useEmissiveLights() ? "1" : "0");
    defines.add("USE_CURVES", scene && (scene->hasGeometryType(Scene::GeometryType::Curve)) ? "1" : "0");
    defines.add("USE_SDF_GRIDS", scene && scene->hasGeometryType(Scene::GeometryType::SDFGrid) ? "1" : "0");
    defines.add("USE_HAIR_MATERIAL", scene && scene->getMaterialCountByType(MaterialType::Hair) > 0u ? "1" : "0");

    // Hacked in camera keyframe count.
    defines.add("NUM_CAMERA_SAMPLES", std::to_string(owner.mCameraParams.currNumKeyframes));

    // Number of time partitions to scatter into.
    defines.add("NUM_TIME_PARTITIONS", std::to_string(owner.mReSTIRParams.numTimePartitions));

    return defines;
}
