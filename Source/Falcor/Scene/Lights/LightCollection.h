/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
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
#include "MeshLightData.slang"
#include "ILightCollection.h"
#include "Core/Macros.h"
#include "Core/Object.h"
#include "Core/API/Buffer.h"
#include "Core/API/Sampler.h"
#include "Core/API/Fence.h"
#include "Core/State/GraphicsState.h"
#include "Core/Program/Program.h"
#include "Core/Program/ProgramVars.h"
#include "Core/Pass/ComputePass.h"
#include "Utils/Math/Vector.h"
#include <memory>
#include <vector>

namespace Falcor
{
    class Scene;
    class RenderContext;
    struct ShaderVar;

    /** Class that holds a collection of mesh lights for a scene.

        Each mesh light is represented by a mesh instance with an emissive material.

        This class has utility functions for updating and pre-processing the mesh lights.
        The LightCollection can be used standalone, but more commonly it will be wrapped
        by an emissive light sampler.
    */
    class FALCOR_API LightCollection : public ILightCollection
    {
        FALCOR_OBJECT(LightCollection)
    public:

        /** Creates a light collection for the given scene.
            Note that update() must be called before the collection is ready to use.
            \param[in] pDevice GPU device.
            \param[in] pRenderContext The render context.
            \param[in] pScene The scene.
            \return A pointer to a new light collection object, or throws an exception if creation failed.
        */
        static ref<LightCollection> create(ref<Device> pDevice, RenderContext* pRenderContext, Scene* pScene)
        {
            return make_ref<LightCollection>(pDevice, pRenderContext, pScene);
        }

        LightCollection(ref<Device> pDevice, RenderContext* pRenderContext, Scene* pScene);
        ~LightCollection() = default;

        const ref<Device>& getDevice() const override { return mpDevice; }

        /** Updates the light collection to the current state of the scene.
            \param[in] pRenderContext The render context.
            \param[out] pUpdateStatus Stores information about which type of updates were performed for each mesh light. This is an optional output parameter.
            \return True if the lighting in the scene has changed since the last frame.
        */
        bool update(RenderContext* pRenderContext, UpdateStatus* pUpdateStatus = nullptr) override;

        /** Bind the light collection data to a given shader var
            \param[in] var The shader variable to set the data into.
        */
        void bindShaderData(const ShaderVar& var) const override;

        /** Returns the total number of triangle lights (may include culled triangles).
        */
        uint32_t getTotalLightCount() const override  { return mTriangleCount; }

        /** Returns stats.
        */
        const MeshLightStats& getStats(RenderContext* pRenderContext) const override { computeStats(pRenderContext); return mMeshLightStats; }

        /** Returns a CPU buffer with all emissive triangles in world space.
            Note that update() must have been called before for the data to be valid.
            Call prepareSyncCPUData() ahead of time to avoid stalling the GPU.
        */
        const std::vector<MeshLightTriangle>& getMeshLightTriangles(RenderContext* pRenderContext) const override { syncCPUData(pRenderContext); return mMeshLightTriangles; }

        /** Returns a CPU buffer with all mesh lights.
            Note that update() must have been called before for the data to be valid.
        */
        const std::vector<MeshLightData>& getMeshLights() const override { return mMeshLights; }

        /** Prepare for syncing the CPU data.
            If the mesh light triangles will be accessed with getMeshLightTriangles()
            performance can be improved by calling this function ahead of time.
            This function schedules the copies so that it can be read back without delay later.
        */
        void prepareSyncCPUData(RenderContext* pRenderContext) const override { copyDataToStagingBuffer(pRenderContext); }

        /** Get the total GPU memory usage in bytes.
        */
        uint64_t getMemoryUsageInBytes() const override;

        // Internal update flags. This only public for FALCOR_ENUM_CLASS_OPERATORS() to work.
        enum class CPUOutOfDateFlags : uint32_t
        {
            None         = 0,
            TriangleData = 0x1,
            FluxData     = 0x2,

            All          = TriangleData | FluxData
        };

        /** Gets a signal interface that is signaled when the LightCollection is updated.
         */
        UpdateFlagsSignal::Interface getUpdateFlagsSignal() override { return mUpdateFlagsSignal.getInterface(); }

    protected:
        void initIntegrator(RenderContext* pRenderContext, const Scene& scene);
        void setupMeshLights(const Scene& scene);
        void build(RenderContext* pRenderContext, const Scene& scene);
        void prepareTriangleData(RenderContext* pRenderContext, const Scene& scene);
        void prepareMeshData(const Scene& scene);
        void integrateEmissive(RenderContext* pRenderContext, const Scene& scene);
        void computeStats(RenderContext* pRenderContext) const;
        void buildTriangleList(RenderContext* pRenderContext, const Scene& scene);
        void updateActiveTriangleList(RenderContext* pRenderContext);
        void updateTrianglePositions(RenderContext* pRenderContext, const Scene& scene, const std::vector<uint32_t>& updatedLights);

        void copyDataToStagingBuffer(RenderContext* pRenderContext) const;
        void syncCPUData(RenderContext* pRenderContext) const;

        // Internal state
        ref<Device>                             mpDevice;
        Scene*                                  mpScene;                ///< Unowning pointer to scene (scene owns LightCollection).

        std::vector<MeshLightData>              mMeshLights;            ///< List of all mesh lights.
        uint32_t                                mTriangleCount = 0;     ///< Total number of triangles in all mesh lights (= mMeshLightTriangles.size()). This may include culled triangles.
        ref<Buffer> mpSceneMeshPrimIDList; ///< Per-triangle lookup data for which scene mesh ID and primitive ID they are>  (mTriangleCount elements)

        mutable std::vector<MeshLightTriangle>  mMeshLightTriangles;    ///< List of all pre-processed mesh light triangles.
        mutable std::vector<uint32_t>           mActiveTriangleList;    ///< List of active (non-culled) emissive triangles.
        mutable std::vector<uint32_t>           mTriToActiveList;       ///< Mapping of all light triangles to index in mActiveTriangleList.

        mutable MeshLightStats                  mMeshLightStats;        ///< Stats before/after pre-processing of mesh lights. Do not access this directly, use getStats() which ensures the stats are up-to-date.
        mutable bool                            mStatsValid = false;    ///< True when stats are valid.

        // GPU resources for the mesh lights and emissive triangles.
        ref<Buffer>                             mpTriangleData;         ///< Per-triangle geometry data for emissive triangles (mTriangleCount elements).
        ref<Buffer>                             mpActiveTriangleList;   ///< List of active (non-culled) emissive triangle.
        ref<Buffer>                             mpTriToActiveList;      ///< Mapping of all light triangles to index in mActiveTriangleList.
        ref<Buffer>                             mpFluxData;             ///< Per-triangle flux data for emissive triangles (mTriangleCount elements).
        ref<Buffer>                             mpMeshData;             ///< Per-mesh data for emissive meshes (mMeshLights.size() elements).
        ref<Buffer>                             mpPerMeshInstanceOffset; ///< Per-mesh instance offset into emissive triangles array (Scene::getMeshInstanceCount() elements).

        mutable ref<Buffer>                     mpStagingBuffer;        ///< Staging buffer used for retrieving the vertex positions, texture coordinates and light IDs from the GPU.
        ref<Fence>                              mpStagingFence;         ///< Fence used for waiting on the staging buffer being filled in.

        ref<Sampler>                            mpSamplerState;         ///< Material sampler for emissive textures.

        // Shader programs.
        struct
        {
            ref<Program>                        pProgram;
            ref<ProgramVars>                    pVars;
            ref<GraphicsState>                  pState;
            ref<Sampler>                        pPointSampler;      ///< Point sampler for fetching individual texels in integrator. Must use same wrap mode etc. as material sampler.
            ref<Buffer>                         pResultBuffer;      ///< The output of the integration pass is written here. Using raw buffer for fp32 compatibility.
        } mIntegrator;

        ref<ComputePass>                        mpTriangleListBuilder;
        ref<ComputePass>                        mpTrianglePositionUpdater;
        ref<ComputePass>                        mpFinalizeIntegration;

        mutable CPUOutOfDateFlags               mCPUInvalidData = CPUOutOfDateFlags::None;  ///< Flags indicating which CPU data is valid.
        mutable bool                            mStagingBufferValid = true;                 ///< Flag to indicate if the contents of the staging buffer is up-to-date.

        UpdateFlagsSignal mUpdateFlagsSignal;
    };

    FALCOR_ENUM_CLASS_OPERATORS(LightCollection::CPUOutOfDateFlags);
}
