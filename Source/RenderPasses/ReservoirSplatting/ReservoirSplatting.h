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

#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "Utils/Debug/PixelDebug.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "Rendering/Utils/PixelStats.h"

#include "Params.slang"
#include "ShiftOptions.slang"

using namespace Falcor;

/** Fast path tracer.
*/
class ReservoirSplatting : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReservoirSplatting, "ReservoirSplatting", "Path Tracer with Reservoir Splatting");

    static ref<ReservoirSplatting> create(ref<Device> pDevice, const Properties& props) { return make_ref<ReservoirSplatting>(pDevice, props); }

    ReservoirSplatting(ref<Device> pDevice, const Properties& props);

    virtual void setProperties(const Properties& props) override;
    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override;
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

    PixelStats& getPixelStats() { return *mpPixelStats; }

    void reset();

    static void registerBindings(pybind11::module& m);

private:
    struct TracePass
    {
        std::string name;
        std::string passDefine;
        ref<Program> pProgram;
        ref<RtBindingTable> pBindingTable;
        ref<RtProgramVars> pVars;

        TracePass(ref<Device> pDevice, const std::string& filename, const std::string& name, const std::string& passDefine, const ref<Scene>& pScene, const DefineList& defines, const TypeConformanceList& globalTypeConformances);
        void prepareProgram(ref<Device> pDevice, const DefineList& defines);
    };

    void parseProperties(const Properties& props);
    void validateOptions();
    void resetPrograms();
    void updatePrograms();
    void setFrameDim(const uint2 frameDim);
    ref<Texture> createNeighborOffsetTexture(const uint32_t sampleCount);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePathTracer(const RenderData& renderData);
    void prepareCameraManager(const RenderData& renderData, bool shouldReset);
    void resetLighting();
    void prepareMaterials(RenderContext* pRenderContext);
    bool prepareLighting(RenderContext* pRenderContext);
    void bindShaderData(const ShaderVar& var, const RenderData& renderData, bool useLightSampling = true) const;
    bool renderScatterVisualizationUI(Gui::Widgets& widget);
    bool renderReSTIRUI(Gui::Widgets& widget);
    bool renderRenderingUI(Gui::Widgets& widget);
    bool renderDebugUI(Gui::Widgets& widget);
    void renderStatsUI(Gui::Widgets& widget);
    bool beginFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void endFrame(RenderContext* pRenderContext, const RenderData& renderData);
    void tracePass(RenderContext* pRenderContext, const RenderData& renderData, TracePass& tracePass, uint z = 1);
    void initialCandidateGeneration(RenderContext* pRenderContext, const RenderData& renderData);
    void robustReuseOptimization(RenderContext* pRenderContext, const RenderData& renderData);
    void collectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData);
    void reprojectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData);
    void multiReprojectTemporalSamples(RenderContext* pRenderContext, const RenderData& renderData);
    void sortReprojectedReservoirs(RenderContext* pRenderContext, const RenderData& renderData);
    void multiSortReprojectedReservoirs(RenderContext* pRenderContext, const RenderData& renderData);
    void visualizeForwardReprojection(RenderContext* pRenderContext, const RenderData& renderData);
    void prepareGatherData(RenderContext* pRenderContext, const RenderData& renderData);
    void gatherTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData);
    void scatterTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData);
    void multiScatterTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData);
    void scatterBackupTemporalResampling(RenderContext* pRenderContext, const RenderData& renderData);
    void spatialResampling(RenderContext* pRenderContext, const RenderData& renderData);
    void resolveReSTIR(RenderContext* pRenderContext, const RenderData& renderData);

    /** Static configuration. Changing any of these options require shader recompilation.
    */
    struct StaticParams
    {
        // Rendering parameters
        uint32_t    samplesPerPixel = 1;                        ///< Number of samples (paths) per pixel, unless a sample density map is used.
        uint32_t    maxSurfaceBounces = 9;                      ///< Max number of surface bounces (diffuse + specular + transmission), up to kMaxPathLength. This will be initialized at startup.
        uint32_t    maxDiffuseBounces = 9;                      ///< Max number of diffuse bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxSpecularBounces = 9;                     ///< Max number of specular bounces (0 = direct only), up to kMaxBounces.
        uint32_t    maxTransmissionBounces = 9;                 ///< Max number of transmission bounces (0 = none), up to kMaxBounces.

        // Sampling parameters
        uint32_t    sampleGenerator = SAMPLE_GENERATOR_TINY_UNIFORM;                ///< Pseudorandom sample generator type.
        bool        useRussianRoulette = true;                                      ///< Use russian roulette to terminate low throughput paths.
        MISHeuristic misHeuristic = MISHeuristic::Balance;                          ///< MIS heuristic.
        float       misPowerExponent = 2.f;                                         ///< MIS exponent for the power heuristic. This is only used when 'PowerExp' is chosen.
        EmissiveLightSamplerType emissiveSampler = EmissiveLightSamplerType::Power; ///< Emissive light sampler to use for NEE. The LightBVH is (potentially) position dependent, so it should be avoided.

        // Material parameters
        bool        useAlphaTest = true;                        ///< Use alpha testing on non-opaque triangles.
        bool        adjustShadingNormals = false;               ///< Adjust shading normals on secondary hits.
        uint32_t    maxNestedMaterials = 2;                     ///< Maximum supported number of nested materials.
        bool        useLightsInDielectricVolumes = false;       ///< Use lights inside of volumes (transmissive materials). We typically don't want this because lights are occluded by the interface.
        bool        disableCaustics = false;                    ///< Disable sampling of caustics.
        TexLODMode  primaryLodMode = TexLODMode::Mip0;          ///< Use filtered texture lookups at the primary hit.

        // Output parameters
        ColorFormat colorFormat = ColorFormat::LogLuvHDR;       ///< Color format used for internal per-sample color and denoiser buffers.

        DefineList getDefines(const ReservoirSplatting& owner) const;
    };

    /** Changes in camera parameters generally don't need shader recompilation.
    */
    struct CameraParams
    {
        // Common parameters
        float artificialFrameTime = 1.0f / 60.0f;   ///< Artificial frame time between the two camera points.

        // Free camera motion parameters.
        int currNumKeyframes = 1;                   ///< Number of camera samples that need to be kept track of.
        int prevNumKeyframes = 0;                   ///< The previous number of camera samples used, in case we need to recompile the shader.
    };

    /** Runtime parameters for ReSTIR.
     */
    struct ReSTIRParams
    {
        // Temporal resampling parameters.
        bool enableTemporalResampling = true;                                   ///< Toggle to use temporal resampling.
        GatherMechanism temporalGathering = GatherMechanism::Fast;              ///< Temporal gathering method (if a gather reuse is used).
        TemporalReuse temporalReuseOption = TemporalReuse::GatherOnly;          ///< Temporal reuse option.
        uint32_t numTimePartitions = 2;                                         ///< Number of time partitions.
        ScatterBackupMIS scatterBackupMISOption = ScatterBackupMIS::Balance;    ///< Hybrid MIS weighting option.
        bool useConfidenceWeightsTemporally = true;                             ///< Toggle for using confidence weights during temporal resampling.

        // Spatial resampling parameters.
        bool enableSpatialResampling = true;                                    ///< Toggle to use spatial resampling.
        uint32_t numSpatialIterations = 1;                                      ///< Number of spatial resampling iterations.
        float spatialGatherRadius = 30.0f;                                      ///< Radius for spatial reuse.
        uint32_t neighborCount = 1;                                             ///< Number of neighbors to consider for spatial resampling (for a given iteration).
        bool useConfidenceWeightsSpatially = true;                              ///< Toggle for using confidence weights during spatial resampling.
    };

    // Configuration
    PathTracerParams                    mParams;                            ///< Runtime path tracer parameters.
    StaticParams                        mStaticParams;                      ///< Static parameters. These are set as compile-time constants in the shaders.
    CameraParams                        mCameraParams;                      ///< Camera parameters that can be adjusted at runtime (for the most part, unless the number of keypoints changes).
    ReSTIRParams                        mReSTIRParams;                      ///< ReSTIR parameters that can be adjusted at runtime.
    mutable LightBVHSampler::Options    mLightBVHOptions;                   ///< Current options for the light BVH sampler.

    bool                                mEnabled = true;                                            ///< Switch to enable/disable the path tracer. When disabled the pass outputs are cleared.
    RenderPassHelpers::IOSize           mOutputSizeSelection = RenderPassHelpers::IOSize::Default;  ///< Selected output size.
    uint2                               mFixedOutputSize = { 512, 512 };                            ///< Output size in pixels when 'Fixed' size is selected.

    // Internal state
    ref<Scene>                              mpScene;                        ///< The current scene, or nullptr if no scene loaded.
    ref<SampleGenerator>                    mpSampleGenerator;              ///< GPU pseudo-random sample generator.
    std::unique_ptr<EnvMapSampler>          mpEnvMapSampler;                ///< Environment map sampler or nullptr if not used.
    std::unique_ptr<EmissiveLightSampler>   mpEmissiveSampler;              ///< Emissive light sampler or nullptr if not used.
    std::unique_ptr<PixelStats>             mpPixelStats;                   ///< Utility class for collecting pixel stats.
    std::unique_ptr<PixelDebug>             mpPixelDebug;                   ///< Utility class for pixel debugging (print in shaders).

    ref<ParameterBlock>             mpPathTracerBlock;                      ///< Parameter block for the path tracer.
    ref<ParameterBlock>             mpCameraManagerBlock;                   ///< Parameter block for the camera manager.
    ref<ParameterBlock>             mpGatherDataBlock;                      ///< Parameter block for the gather data.

    bool                            mRecompile = false;                     ///< Set to true when program specialization has changed.
    bool                            mVarsChanged = true;                    ///< This is set to true whenever the program vars have changed and resources need to be rebound.
    bool                            mOptionsChanged = false;                ///< True if the config has changed since last frame.
    bool                            mGBufferAdjustShadingNormals = false;   ///< True if GBuffer/VBuffer has adjusted shading normals enabled.

    ref<ComputePass>                mpReflectTypes;                         ///< Helper for reflecting structured buffer types.

    // ReSTIR passes
    ref<ComputePass>                mpInitialCandidatesPass;                    ///< Initial candidate generation pass.
    std::unique_ptr<TracePass>      mpRobustReuseOptimizationPass;              ///< Robust re-use optimization pass.
    ref<ComputePass>                mpCollectTemporalSamplesPass;               ///< Collect temporal samples pass.
    std::unique_ptr<TracePass>      mpReprojectTemporalSamplesPass;             ///< Scatter temporal samples pass.
    std::unique_ptr<TracePass>      mpMultiReprojectTemporalSamplesPass;        ///< Multi-scatter temporal samples pass.
    ref<ComputePass>                mpComputeCellOffsetsPass;                   ///< Compute cell offsets pass.
    ref<ComputePass>                mpSortCellDataPass;                         ///< Sort cell data pass.
    ref<ComputePass>                mpMultiComputeCellOffsetsPass;              ///< (Multi) compute cell offsets pass.
    ref<ComputePass>                mpMultiSortCellDataPass;                    ///< (Multi) sort cell data pass.
    std::unique_ptr<TracePass>      mpGatherTemporalResamplingPass;             ///< Gather-only temporal resampling pass.
    std::unique_ptr<TracePass>      mpScatterTemporalResamplingPass;            ///< Scatter-only temporal resampling pass.
    std::unique_ptr<TracePass>      mpMultiScatterTemporalResamplingPass;       ///< Multi-scatter temporal resampling pass.
    std::unique_ptr<TracePass>      mpScatterBackupTemporalResamplingPass;      ///< Scatter + backup temporal resampling.
    std::unique_ptr<TracePass>      mpSpatialResamplingPass;                    ///< Spatial resampling pass.
    ref<ComputePass>                mpResolveReSTIRPass;                        ///< Resolve ReSTIR pass.

    // Visualization passes
    ref<ComputePass>                mpVisualizeForwardReprojectionPass;         ///< Visualize forward reprojection pass.

    // Reservoirs and reconnection data for ReSTIR
    ref<Buffer>                     mpCurrReservoirs;                       ///< The current reservoir stores the canonical sample from the initial candidate generation pass.
    ref<Buffer>                     mpIntermediateReservoirs;               ///< The intermediate reservoir stores the selected sample for temporal reuse.
    ref<Buffer>                     mpPrevReservoirs;                       ///< The previous reservoir stores all the samples from the previous frame.
    ref<Buffer>                     mpCurrReconnectionData;                 ///< The current reconnection data stores the info needed to reconnect into the current domain.
    ref<Buffer>                     mpIntermediateReconnectionData;         ///< The intermediate reconnection data stores the reconnection data from the selected temporal sample.
    ref<Buffer>                     mpPrevReconnectionData;                 ///< The previous reconnection data stores the info needed to reconnect into the previous domain.

    // Gather temporal resampling
    ref<Buffer>                     mpShiftedPaths;                         ///< Shifted integrand + destination Jacobians for robust reuse.
    ref<Texture>                    mpFloatingCoordinates;                  ///< Fractional coordinates of the "floating" reservoir.

    // Scatter temporal resampling
    ref<Buffer>                     mpGlobalCounters;                       ///< The global counters for scattered reservoir indices and cell offsets.
    ref<Buffer>                     mpCellCounters;                         ///< The local cell sizes for each pixel.
    ref<Buffer>                     mpReservoirIndices;                     ///< Which cell a reservoir was scattered into, and its corresponding index in the cell.
    ref<Buffer>                     mpScatteredReservoirs;                  ///< The scattered reservoirs (in terms of pixels from the old frame).
    ref<Buffer>                     mpCellOffsets;                          ///< The beginning index of each cell in a linearized buffer.
    ref<Buffer>                     mpSortedReservoirs;                     ///< Linearized, sorted buffer of all reservoirs.

    // Multi-scatter temporal resampling
    std::vector<ref<Buffer>>        mMultiGlobalCounters;                   ///< The global counters for multiple scattering.
    std::vector<ref<Buffer>>        mMultiCellCounters;                     ///< The local cell sizes for each pixel, per time partition.
    std::vector<ref<Buffer>>        mMultiReservoirIndices;                 ///< Which cell a reservoir was multi-scattered into, and its corresponding index.
    std::vector<ref<Buffer>>        mMultiScatteredReservoirs;              ///< The multi-scattered reservoirs (in terms of pixels from the old frame).
    std::vector<ref<Buffer>>        mMultiCellOffsets;                      ///< The beginning index of each cell in a linearized buffer.
    std::vector<ref<Buffer>>        mMultiSortedReservoirs;                 ///> Linearized, sorted buffer of all reservoirs.

    // Temporal scatter visualization
    bool                            mDumpScatterCount = false;              ///< Dump the scatter count on next run.
    std::string                     mScatterDumpDir = "c:/scatterCount/";   ///< Directory to output the scatter counts.
    std::string                     mScatterDumpFile = "dump.txt";          ///< File to output the scatter counts.
    ref<Fence>                      mpReadbackFence;                        ///< Readback fence for staging.
    ref<Buffer>                     mpTotalCellCounters;                    ///< The total scatter count, including multi-scatter.
    ref<Buffer>                     mpStagingTotalCellCounters;             ///< The total scatter count, including multi-scatter (staging buffer).

    // Spatial resampling
    ref<Texture>                    mpNeighborOffsets;                      ///< Quasi-random numbers to sample neighbor reservoirs.

    // When the scene is frozen, we want to preserve the previous reservoirs and reconnection data,
    // so during the spatial pass, we overwrite temporary buffers instead.
    ref<Buffer>                     mpTempReservoirs;
    ref<Buffer>                     mpTempReconnectionData;
};
