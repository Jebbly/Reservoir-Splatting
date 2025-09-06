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

import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.IsotropicGGX;
import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Rendering.Utils.PixelStats;
import RenderPasses.Shared.Denoising.NRDHelpers;
import Scene.Raytracing;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Utils.Debug.PixelDebug;
import Utils.Math.MathHelpers;
import LoadShadingData;
import ReconnectionData;
import Reservoir;
__exported import PathStates;
__exported import Params;
__exported import CameraManager;
import Scene.RaytracingInline;

// Define ray indices.
static const uint kRayTypeScatter = 0;
static const uint kMissScatter = 0;

static const uint kRayTypeShadow = 1;
static const uint kMissShadow = 1;

/** Path tracer.

    This implements the high-level path tracing logic and is shared among
    different scheduling implementations.
*/
struct PathTracer
{
    PathTracerParams params;                            ///< Runtime parameters.

    // Samplers
    EnvMapSampler envMapSampler;                        ///< Environment map sampler. Only valid when kUseEnvLight == true.
    EmissiveLightSampler emissiveSampler;               ///< Emissive light sampler. Only valid when kUseEmissiveLights == true.

    /*******************************************************************
                                Static members
    *******************************************************************/

    // Render settings that depend on the scene.
    // TODO: Move into scene defines.
    static const bool kUseEnvLight = USE_ENV_LIGHT;
    static const bool kUseEmissiveLights = USE_EMISSIVE_LIGHTS;
    static const bool kUseAnalyticLights = USE_ANALYTIC_LIGHTS;
    static const bool kUseCurves = USE_CURVES;
    static const bool kUseHairMaterial = USE_HAIR_MATERIAL;

    /** Types of samplable lights.
    */
    enum class LightType
    {
        EnvMap,
        Emissive,
        Analytic
    };

    /** Describes a light sample.
    */
    struct LightSample
    {
        float3  Li;         ///< Incident radiance at the shading point (unshadowed). This is already divided by the pdf.
        float   pdf;        ///< Pdf with respect to solid angle at the shading point.
        float3  origin;     ///< Ray origin for visibility evaluation (offseted to avoid self-intersection).
        float   distance;   ///< Ray distance for visibility evaluation (shortened to avoid self-intersection).
        float3  dir;        ///< Ray direction for visibility evaluation (normalized).
        uint lightType;     ///< Light type this sample comes from (LightType casted to uint).
        uint triangleId;    ///< for reconstructing hitInfo for sampled emissive lights
        float2 barycentrics; ///< for reconstructing hitInfo for sampled emissive lights
        float geometryTerm;  ///< for computing jacobian in GRIS
        float3 normal;       ///< normal for area (including sphere) light

        Ray getVisibilityRay() { return Ray(origin, dir, 0.f, distance); }
    };

    /** Describes a path vertex.
    */
    struct PathVertex
    {
        uint index;         ///< Vertex index (0 = camera, 1 = primary hit, 2 = secondary hit, etc.).
        float3 pos;         ///< Vertex position.
        float3 faceNormal;  ///< Geometry normal at the vertex (zero if not on a surface).
        bool frontFacing;   ///< True if path vertex is on the front-facing side (if on a surface).

        /** Initializes a path vertex.
            \param[in] index Vertex index.
            \param[in] pos Vertex position.
            \param[in] faceNormal Geometry normal.
            \param[in] frontFacing Front-facing flag.
        */
        __init(uint index, float3 pos, float3 faceNormal = float3(0.f), bool frontFacing = true)
        {
            this.index = index;
            this.pos = pos;
            this.faceNormal = faceNormal;
            this.frontFacing = frontFacing;
        }

        /** Get position with offset applied in direction of the geometry normal to avoid self-intersection
            for visibility rays.
            \param[in] rayDir Direction of the visibility ray (does not need to be normalized).
            \return Returns the offseted position.
        */
        float3 getRayOrigin(float3 rayDir)
        {
            return computeRayOrigin(pos, dot(faceNormal, rayDir) >= 0 ? faceNormal : -faceNormal);
        }

        /** Returns the oriented face normal.
            \return Face normal flipped to the same side as the view vector.
        */
        float3 getOrientedFaceNormal()
        {
            return frontFacing ? faceNormal : -faceNormal;
        }
    };

    /*******************************************************************
                              Member functions
    *******************************************************************/

    /** Check if the path has finished all surface bounces and needs to be terminated.
        Note: This is expected to be called after generateScatterRay(), which increments the bounce counters.
        \param[in] path Path state.
        \return Returns true if path has processed all bounces.
    */
    bool hasFinishedSurfaceBounces(const PathState path)
    {
        const uint diffuseBounces = path.getBounces(BounceType::Diffuse);
        const uint specularBounces = path.getBounces(BounceType::Specular);
        const uint transmissionBounces = path.getBounces(BounceType::Transmission);
        const uint surfaceBounces = diffuseBounces + specularBounces + transmissionBounces;
        return
            (surfaceBounces > kMaxSurfaceBounces) ||
            (diffuseBounces > kMaxDiffuseBounces) ||
            (specularBounces > kMaxSpecularBounces) ||
            (transmissionBounces > kMaxTransmissionBounces);
    }

    /** Compute the total length of a terminated path.
        \param[in] path Path state.
        \return Returns the total number of bounces a path took.
    */
    uint getTerminatedPathLength(const PathState path)
    {
        // Account for the fact that we may have counted one bounce too many (scatter ray at the last path vertex).
        uint diffuseBounces = min(kMaxDiffuseBounces, path.getBounces(BounceType::Diffuse));
        uint specularBounces = min(kMaxSpecularBounces, path.getBounces(BounceType::Specular));
        uint transmissionBounces = min(kMaxTransmissionBounces, path.getBounces(BounceType::Transmission));
        uint surfaceBounces = min(kMaxSurfaceBounces, diffuseBounces + specularBounces + transmissionBounces);
        return surfaceBounces;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[out] path Path state before the primary hit.
        \param[in] seed RNG seed.
        \param[in] pathID Path ID which encodes pixel and sample index.
    */
    void generatePath(out PathState path, const uint seed, const uint pathID)
    {
        path = {};
        path.setActive();
        path.id = pathID;
        path.thp = float3(1.f);

        // Create sample generator.
        const uint2 pixel = path.getPixel();
        path.sg = SampleGenerator(pixel, seed);

        // Randomize the camera parameters.
        Camera camera = gScene.camera;
        path.reconnection.time = sampleNext1D(path.sg) * camera.data.shutterSpeed;
        path.reconnection.subPixel = sampleNext2D(path.sg);
        path.reconnection.lensSample = (camera.data.apertureRadius > 0.0f) ? sample_disk(sampleNext2D(path.sg)) : float2(0.0f, 0.0f);

        // Spawn the primary ray based on the camera orientation.
        gCameraManager.orientCamera(camera, params.frameCount, path.reconnection.time);
        Ray cameraRay = camera.computeRayThinlensSubPixelRandom(params.frameDim, float2(pixel) + path.reconnection.subPixel, path.reconnection.lensSample);

        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        float3 lensLocal = camera.data.apertureRadius * (path.reconnection.lensSample.x * normalize(camera.data.cameraU) +
                                                         path.reconnection.lensSample.y * normalize(camera.data.cameraV));
        path.prevHitPos = camera.data.posW + lensLocal;
    }

    /** Generate the path state for a primary hit in screen space.
        This is treated equivalent to subsequent path vertices to reduce code divergence.
        \param[out] path Path state before the primary hit.
        \param[in] seed RNG seed.
        \param[in] time Timestamp for the path.
        \param[in] fractionalPixel Subpixel of the path.
    */
    void generatePathWithParameters(out PathState path, const uint seed, const float time, const float2 fractionalPixel, const float2 lensSample)
    {
        path = {};
        path.setActive();
        path.thp = float3(1.f);

        // Create sample generator.
        const uint maxSpp = kSamplesPerPixel;
        path.sg = SampleGenerator(uint2(fractionalPixel), seed);

        // Create primary ray.
        Camera camera = gScene.camera;

        // Orient the camera at a given time.
        gCameraManager.orientCamera(camera, params.frameCount, time);
        path.time = time;

        // Spawn the primary ray based on the camera orientation.
        Ray cameraRay = camera.computeRayThinlensSubPixelRandom(params.frameDim, fractionalPixel, lensSample);
        path.origin = cameraRay.origin;
        path.dir = cameraRay.dir;

        float3 lensLocal = camera.data.apertureRadius * (path.reconnection.lensSample.x * normalize(camera.data.cameraU) +
                                                         path.reconnection.lensSample.y * normalize(camera.data.cameraV));
        path.prevHitPos = camera.data.posW + lensLocal;
    }

    struct ClosestHitQuery
    {
        HitInfo traceRay(inout PathState path, const Ray ray, out float hitT)
        {
            SceneRayQuery<kUseAlphaTest> sceneRayQuery;
            return sceneRayQuery.traceRay(ray, hitT);
        }
    };

    struct VisibilityQuery
    {
        bool traceVisibilityRay(const Ray ray)
        {
            SceneRayQuery<kUseAlphaTest> sceneRayQuery;
            return sceneRayQuery.traceVisibilityRay(ray);
        }
    };
    /** Trace a ray to find the next hit.
        This replaces `PathTracer::nextHit` but without support for volume sampling.
        \param[in,out] path Path state.
    */
    void nextHit(inout PathState path)
    {
        // Advance to next path vertex.
        path.incrementVertexIndex();

        // Trace ray.
        logTraceRay(PixelStatsRayType::ClosestHit);
        const Ray ray = path.getScatterRay();

        ClosestHitQuery chq;
        float hitT;
        HitInfo hit = chq.traceRay(path, ray, hitT);

        if (hit.isValid())
        {
            path.setHit(hit);
        }
        else
        {
            path.clearHit();
        }
    }

    /** Set up path for logging and debugging.
        \param[in] path Path state.
    */
    void setupPathLogging(const PathState path)
    {
        printSetPixel(path.getPixel());
        logSetPixel(path.getPixel());
    }

    /** Generates a new scatter ray using BSDF importance sampling, without any special lobe-sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateGuideRay(const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;
        bool valid = mi.sample(sd, path.sg, result, true);
        if (valid) valid = generateScatterRay(result, sd, mi, path);

        return valid;
    }

    /** Generates a new scatter ray using BSDF importance sampling.
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(inout ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        BSDFSample result;

        // Set the lobe mask to a non-zero value, so that the PDF is only of the sampled lobe.
        sd.mtl.setLobeMask(1);
        bool valid = mi.sample(sd, path.sg, result, true);
        path.pdf = result.pdf; // Lobe-specific PDF.

        uint componentType = 0;
        if (result.isLobe(LobeType::DiffuseReflection))
        {
            if (result.isLobe(LobeType::Specular) || result.isLobe(LobeType::Delta))
            {
                componentType = 3;
                result.lobeType &= ~((uint)LobeType::DiffuseReflection);
            }
        }
        else
        {
            if (result.isLobe(LobeType::DiffuseTransmission))
                componentType = 1;
            else if (result.isLobe(LobeType::Specular) || result.isLobe(LobeType::Delta))
                componentType = 2;
        }

        // If we're at the first or second vertex, we care about which lobe was sampled.
        if (path.getVertexIndex() == 1)
        {
            path.reconnection.firstBSDFComponentType = componentType;
        }
        else if (path.getVertexIndex() == 2)
        {
            path.reconnection.secondBSDFComponentType = componentType;
            path.reconnection.secondWo = result.wo;
        }

        // Set the lobe mask back to 0 to get the total PDF.
        sd.mtl.setLobeMask(0xF);
        path.totalPdf = mi.evalPdf(sd, result.wo, true);

        if (valid) generateScatterRay({result.wo, result.pdf, result.weight, result.lobeType}, sd, mi, path);

        // We reset the throguhput at the first two vertices.
        if (path.getVertexIndex() == 1 || path.getVertexIndex() == 2)
        {
            path.reconnection.earlyThroughput *= result.weight;
            path.thp = float3(1.0f, 1.0f, 1.0f);
        }

        return valid;
    }

    /** Generates a new scatter ray given a valid BSDF sample.
        \param[in] bs BSDF sample (assumed to be valid).
        \param[in] sd Shading data.
        \param[in] mi Material instance at the shading point.
        \param[in,out] path The path state.
        \return True if a ray was generated, false otherwise.
    */
    bool generateScatterRay(const BSDFSample bs, const ShadingData sd, const IMaterialInstance mi, inout PathState path)
    {
        const bool isTriangleHit = path.hit.getType() == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (path.hit.getType() == HitType::Curve);
        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        path.dir = bs.wo;
        path.thp *= bs.weight;

        path.clearEventFlags();

        // Handle reflection events.
        if (bs.isLobe(LobeType::Reflection))
        {
            // We classify specular events as diffuse if the roughness is above some threshold.
            float roughness = mi.getProperties(sd).roughness;
            bool isDiffuse = bs.isLobe(LobeType::DiffuseReflection) || roughness > params.specularRoughnessThreshold;

            if (isDiffuse)
            {
                path.incrementBounces(BounceType::Diffuse);
            }
            else
            {
                path.incrementBounces(BounceType::Specular);
                path.setSpecular();
            }
        }

        // Handle delta events.
        if (bs.isLobe(LobeType::Delta))
        {
            path.setDelta();
        }

        // Handle transmission events.
        if (bs.isLobe(LobeType::Transmission))
        {
            path.incrementBounces(BounceType::Transmission);
            path.setTransmission();

            if (path.getVertexIndex() == 1)
            {
                path.reconnection.transmissionEvent = true;
            }

            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.frame.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }
        }

        // Save the normal used for NEE. This is needed for MIS.
        path.normal = sd.getOrientedFaceNormal();

        // Mark the path as valid only if it has a non-zero throughput.
        bool valid = any(path.thp > 0.f);

        return valid;
    }

    /** Evaluates the currently configured heuristic for multiple importance sampling (MIS).
        \param[in] n0 Number of samples taken from the first sampling strategy.
        \param[in] p0 Pdf for the first sampling strategy.
        \param[in] n1 Number of samples taken from the second sampling strategy.
        \param[in] p1 Pdf for the second sampling strategy.
        \return Weight for the contribution from the first strategy (p0).
    */
    static float evalMIS(float n0, float p0, float n1, float p1)
    {
        switch (MISHeuristic(kMISHeuristic))
        {
        case MISHeuristic::Balance:
        {
            // Balance heuristic
            float q0 = n0 * p0;
            float q1 = n1 * p1;
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerTwo:
        {
            // Power two heuristic
            float q0 = (n0 * p0) * (n0 * p0);
            float q1 = (n1 * p1) * (n1 * p1);
            return q0 / (q0 + q1);
        }
        case MISHeuristic::PowerExp:
        {
            // Power exp heuristic
            float q0 = pow(n0 * p0, kMISPowerExponent);
            float q1 = pow(n1 * p1, kMISPowerExponent);
            return q0 / (q0 + q1);
        }
        default:
            return 0.f;
        }
    }

    /** Generates a light sample on the environment map.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEnvMapSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        if (!kUseEnvLight) return false;

        // Sample environment map.
        EnvMapSample lightSample;
        if (!envMapSampler.sample(sampleNext2D(sg), lightSample)) return false;

        // Setup returned sample.
        ls.Li = lightSample.pdf > 0.f ? lightSample.Le / lightSample.pdf : float3(0);
        ls.pdf = lightSample.pdf;
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        ls.distance = kRayTMax;
        ls.dir = lightSample.dir;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the emissive geometry.
        \param[in] vertex Path vertex.
        \param[in] upperHemisphere True if only upper hemisphere should be considered.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateEmissiveSample(const PathVertex vertex, const bool upperHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.
        if (!kUseEmissiveLights) return false;

        TriangleLightSample tls;
        if (!emissiveSampler.sampleLight(vertex.pos, vertex.getOrientedFaceNormal(), upperHemisphere, sg, tls)) return false;

        // Setup returned sample.
        ls.Li = tls.pdf > 0.f ? tls.Le / tls.pdf : float3(0);
        ls.pdf = tls.pdf;
        // Offset shading and light position to avoid self-intersection.
        float3 lightPos = computeRayOrigin(tls.posW, tls.normalW);
        ls.origin = vertex.getRayOrigin(lightPos - vertex.pos);
        float3 toLight = lightPos - ls.origin;
        ls.distance = length(toLight);
        ls.dir = normalize(toLight);
        ls.barycentrics = tls.uv;
        ls.triangleId = tls.triangleIndex;
        ls.geometryTerm = tls.geometryTerm;

        return any(ls.Li > 0.f);
    }

    /** Generates a light sample on the analytic lights.
        \param[in] vertex Path vertex.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateAnalyticLightSample(const PathVertex vertex, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {}; // Default initialization to avoid divergence at returns.

        uint lightCount = gScene.getLightCount();
        if (!kUseAnalyticLights || lightCount == 0) return false;

        // Sample analytic light source selected uniformly from the light list.
        // TODO: Sample based on estimated contributions as pdf.
        uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1);

        // Sample local light source.
        AnalyticLightSample lightSample;
        if (!sampleLight(vertex.pos, gScene.getLight(lightIndex), sg, lightSample)) return false;

        // Setup returned sample.
        ls.pdf = lightSample.pdf / lightCount;
        ls.Li = lightSample.Li * lightCount;
        // Offset shading position to avoid self-intersection.
        ls.origin = vertex.getRayOrigin(lightSample.dir);
        // Analytic lights do not currently have a geometric representation in the scene.
        // Do not worry about adjusting the ray length to avoid self-intersections at the light.
        ls.distance = lightSample.distance;
        ls.dir = lightSample.dir;
        ls.normal = lightSample.normalW;
        ls.geometryTerm = ls.pdf == 0.f ? 1.f / (ls.distance * ls.distance) : abs(dot(ls.normal, ls.dir)) / (ls.distance * ls.distance);

        return any(ls.Li > 0.f);
    }

    /** Return the probabilities for selecting different light types.
        \param[out] p Probabilities.
    */
    void getLightTypeSelectionProbabilities(out float p[3])
    {
        // Set relative probabilities of the different sampling techniques.
        // TODO: These should use estimated irradiance from each light type. Using equal probabilities for now.
        p[0] = kUseEnvLight ? 1.f : 0.f;
        p[1] = kUseEmissiveLights ? 1.f : 0.f;
        p[2] = kUseAnalyticLights ? 1.f : 0.f;

        // Normalize probabilities. Early out if zero.
        float sum = p[0] + p[1] + p[2];
        if (sum == 0.f) return;

        float invSum = 1.f / sum;
        p[0] *= invSum;
        p[1] *= invSum;
        p[2] *= invSum;
    }

    float getEnvMapSelectionProbability()   { float p[3]; getLightTypeSelectionProbabilities(p); return p[0]; }
    float getEmissiveSelectionProbability() { float p[3]; getLightTypeSelectionProbabilities(p); return p[1]; }
    float getAnalyticSelectionProbability()  { float p[3]; getLightTypeSelectionProbabilities(p); return p[2]; }

    /** Select a light type for sampling.
        \param[out] lightType Selected light type.
        \param[out] pdf Probability for selected type.
        \param[in,out] sg Sample generator.
        \return Return true if selection is valid.
    */
    bool selectLightType(out uint lightType, out float pdf, inout SampleGenerator sg)
    {
        float p[3];
        getLightTypeSelectionProbabilities(p);

        float u = sampleNext1D(sg);

        [unroll]
        for (lightType = 0; lightType < 3; ++lightType)
        {
            if (u < p[lightType])
            {
                pdf = p[lightType];
                return true;
            }
            u -= p[lightType];
        }

        lightType = {};
        pdf = {};

        return false;
    }

    /** Samples a light source in the scene.
        This function first stochastically selects a type of light source to sample,
        and then calls that the sampling function for the chosen light type.
        \param[in] vertex Path vertex.
        \param[in] sampleUpperHemisphere True if the upper hemisphere should be sampled.
        \param[in] sampleLowerHemisphere True if the lower hemisphere should be sampled.
        \param[in,out] sg Sample generator.
        \param[out] ls Struct describing valid samples.
        \return True if the sample is valid and has nonzero contribution, false otherwise.
    */
    bool generateLightSample(const PathVertex vertex, const bool sampleUpperHemisphere, const bool sampleLowerHemisphere, inout SampleGenerator sg, out LightSample ls)
    {
        ls = {};

        uint lightType;
        float selectionPdf;
        if (!selectLightType(lightType, selectionPdf, sg)) return false;

        bool valid = false;
        if (kUseEnvLight && lightType == (uint)LightType::EnvMap) valid = generateEnvMapSample(vertex, sg, ls);
        if (kUseEmissiveLights && lightType == (uint)LightType::Emissive)
        {
            // Emissive light samplers have an option to exclusively sample the upper hemisphere.
            bool upperHemisphere = sampleUpperHemisphere && !sampleLowerHemisphere;
            valid = generateEmissiveSample(vertex, upperHemisphere, sg, ls);
        }
        if (kUseAnalyticLights && lightType == (uint)LightType::Analytic)
        {
            valid = generateAnalyticLightSample(vertex, sg, ls);
        }
        if (!valid) return false;

        // Reject samples in non-requested hemispheres.
        float NdotL = dot(vertex.getOrientedFaceNormal(), ls.dir);
        if ((!sampleUpperHemisphere && NdotL >= -kMinCosTheta) || (!sampleLowerHemisphere && NdotL <= kMinCosTheta))
            return false;

        // Account for light type selection.
        ls.lightType = lightType;
        ls.pdf *= selectionPdf;
        ls.Li /= selectionPdf;

        return true;
    }

    /** Handle hits on dielectrics.
        \return True if this is an valid intersection, false if it is rejected.
    */
    bool handleNestedDielectrics(inout ShadingData sd, inout PathState path)
    {
        // Check for false intersections.
        uint nestedPriority = sd.mtl.getNestedPriority();
        if (!path.interiorList.isTrueIntersection(nestedPriority))
        {
            // If it is a false intersection, we reject the hit and continue the path
            // on the other side of the interface.
            // If the offset position is not quite large enough, due to self-intersections
            // it is possible we repeatedly hit the same surface and try to reject it.
            // This has happened in a few occasions with large transmissive triangles.
            // As a workaround, count number of rejected hits and terminate the path if too many.
            if (path.rejectedHits < kMaxRejectedHits)
            {
                path.rejectedHits++;
                path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                path.origin = sd.computeRayOrigin(false);
                path.decrementVertexIndex();
            }
            else
            {
                path.terminate();
            }
            return false;
        }

        // Compute index of refraction for medium on the outside.
        sd.IoR = computeOutsideIoR(path.interiorList, sd.materialID, sd.frontFacing);

        return true;
    }

    /** Apply russian roulette to terminate paths early.
        \param[in,out] path Path.
        \param[in] u Uniform random number in [0,1).
        \return Returns true if path was terminated.
    */
    bool terminatePathByRussianRoulette(inout PathState path, float u)
    {
        const float rrVal = luminance(path.thp);
        const float prob = max(0.f, 1.f - rrVal);
        if (u < prob)
        {
            path.terminate();
            return true;
        }
        path.thp /= 1.f - prob;
        return false;
    }

    /** Helper to create a texture sampler instance.
        The method for computing texture level-of-detail depends on the configuration.
        \param[in] path Path state.
        \param[in] isPrimaryTriangleHit True if primary hit on a triangle.
        \return Texture sampler instance.
    */
    ITextureSampler createTextureSampler(const PathState path, bool isPrimaryHit, bool isTriangleHit)
    {
        if (kPrimaryLodMode == TexLODMode::RayDiffs && isPrimaryHit && isTriangleHit)
        {
            // Filtered lookups at primary hit on triangle.
            float2 ddx, ddy;
            computeDerivativesAtPrimaryTriangleHit(path.hit.getTriangleHit(), path.getPixel(), params.frameDim, ddx, ddy);
            return ExplicitGradientTextureSampler(ddx, ddy);
        }
        else
        {
            float lod = isPrimaryHit ? 0.f : params.lodBias;
            return ExplicitLodTextureSampler(lod);
        }
    }

    /** Handle the case when a scatter ray hits a surface.
        After handling the hit, a new scatter ray is generated or the path is terminated.
        \param[in,out] path The path state.
        \return The true position of the hit.
    */
    float3 handleHit(inout PathState path)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Sample light(s) using shadow rays
        // - Sample scatter ray or terminate

        setupPathLogging(path);

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);

        float distanceToPrevVertex = length(sd.posW - path.prevHitPos);
        float geometryFactor = abs(dot(sd.faceN, sd.V)) / (distanceToPrevVertex * distanceToPrevVertex);
        float prevScatterPdf = path.pdf;
        path.prevHitPos = sd.posW;

        // We can update common reconnection data at certain points.
        if (path.getVertexIndex() == 1)
        {
            // print("First Hit", sd.posW);
            path.reconnection.firstHit = path.hit;
            path.reconnection.firstWi = -path.dir;

            Camera camera = gScene.camera;
            gCameraManager.orientCamera(camera, params.frameCount, path.time);

            float3 lensLocal = camera.data.apertureRadius * (path.reconnection.lensSample.x * normalize(camera.data.cameraU) +
                                                             path.reconnection.lensSample.y * normalize(camera.data.cameraV));
            float3 x0 = camera.data.posW + lensLocal;

            float dist = length(sd.posW - x0);
            float distSquared = dist * dist;
            float cosNormal = dot(-path.dir, sd.faceN);
            float cosSensor = dot(normalize(camera.data.cameraW), path.dir);

            path.reconnection.subPixelJacobian = abs(cosNormal / distSquared) / abs(pow(cosSensor, 3));

            // Also hold onto the lens vertex Jacobian
            float3 camDir = normalize(camera.data.cameraW);
            float camZ = dot(sd.posW - camera.data.posW, camDir);

            float d0 = (length(camera.data.cameraW) / abs(camZ) * dist);
            float d1 = dist - d0;

            path.reconnection.lensVertexJacobian = (d0 * d0) / (d1 * d1) * (abs(dot(-path.dir, sd.faceN))) / (abs(dot(-path.dir, camDir)));
        }
        else if (path.getVertexIndex() == 2)
        {
            path.reconnection.secondHit = path.hit;
            path.reconnection.secondaryPathJacobian = prevScatterPdf * geometryFactor;

            // print("Second Hit Normal", sd.faceN);
            // print("Second Hit Incoming Dir", sd.V);
            // print("Second Hit Position", sd.posW);
            // print("Distance", distanceToPrevVertex);
            // print("Cos Theta", abs(dot(sd.faceN, sd.V)));
            // print("Geometry Factor", geometryFactor);
            // print("First Scatter Pdf", prevScatterPdf);
        }
        else if (path.getVertexIndex() == 3)
        {
            path.reconnection.secondaryPathJacobian *= prevScatterPdf;
            // print("Second Scatter Pdf", prevScatterPdf);
        }

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            path.thp *= HomogeneousVolumeSampler::evalTransmittance(vp, length(sd.posW - path.origin));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return sd.posW;

        logPathVertex();

        // Create BSDF instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Optionally disable emission inside volumes.
        if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume())
        {
            bsdfProperties.emission = float3(0.f);
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add emitted radiance.
        // The primary hit is always included, secondary hits only if emissive lights are enabled and the full light contribution hasn't been sampled elsewhere.
        bool computeEmissive = isPrimaryHit || kUseEmissiveLights && (!path.isLightSampled() || !isLightSamplable);

        float3 attenuatedEmission = 0.f;
        if (computeEmissive && any(bsdfProperties.emission > 0.f))
        {
            float misWeight = 1.f;
            float lightPdf = 0.0f;
            if (kUseEmissiveLights && isTriangleHit && !isPrimaryHit && path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled emissive lights,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.
                // Note that MIS is only applied for hits on emissive triangles (other emissive geometry is not supported).

                // Prepare hit point struct with data needed for emissive light PDF evaluation.
                TriangleHit triangleHit = path.hit.getTriangleHit();
                TriangleLightHit hit;
                hit.triangleIndex = gScene.lightCollection.getTriangleIndex(triangleHit.instanceID, triangleHit.primitiveIndex);
                hit.posW = sd.posW;
                hit.normalW = sd.getOrientedFaceNormal();

                // Evaluate PDF at the hit, had it been generated with light sampling.
                // Emissive light samplers have an option to exclusively sample the upper hemisphere.
                bool upperHemisphere = path.isLightSampledUpper() && !path.isLightSampledLower();
                lightPdf = getEmissiveSelectionProbability() * emissiveSampler.evalPdf(path.origin, path.normal, upperHemisphere, hit);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.totalPdf, 1, lightPdf);
            }

            // Accumulate emitted radiance weighted by path throughput and MIS weight.
            attenuatedEmission = path.thp * misWeight * bsdfProperties.emission;

            bool selected = path.reservoir.addVertex(path.sg, 1, path.reconnection.earlyThroughput * path.thp * bsdfProperties.emission * misWeight);
            if (selected)
            {
                // Copy over any relevant accumulated data (e.g. the Jacobian).
                path.reservoir.selectedReconnection = path.reconnection;

                path.reservoir.selectedReconnection.pathLength = path.getVertexIndex();
                path.reservoir.selectedReconnection.lightIsNEE = false;
                path.reservoir.selectedReconnection.lightIsDistant = false;
                path.reservoir.selectedReconnection.lightPdf = lightPdf;

                // print("BSDF", path.reservoir.selectedReconnection.pathLength);
                // print("MIS Weight", misWeight);
                // print("Early Throughput", path.reconnection.earlyThroughput);
                // print("Path Throughput", path.thp);
                // print("Light BSDF Emission", bsdfProperties.emission);
                // print("Path Contribution", path.reconnection.earlyThroughput * path.thp * bsdfProperties.emission * misWeight);

                if (path.getVertexIndex() == 2)
                {
                    // Converts solid angle measure to area measure.
                    path.reservoir.selectedReconnection.lightPdf *= geometryFactor;
                }

                // if the path length is the second or third vertex, we need to re-evaluate the MIS weight - otherwise, include it.
                // Here, path.thp = float3(1) if it's the first or second vertex.
                if (path.getVertexIndex() == 2 || path.getVertexIndex() == 3)
                {
                    path.reservoir.selectedReconnection.irradiance = bsdfProperties.emission;
                }
                else
                {
                    path.reservoir.selectedReconnection.irradiance = path.thp * bsdfProperties.emission * misWeight;
                }
            }
        }

        // Terminate after scatter ray on last vertex has been processed.
        if (hasFinishedSurfaceBounces(path))
        {
            path.terminate();
            return sd.posW;
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.frame.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeRayOrigin();
        }

        // Determine if BSDF has non-delta lobes.
        const uint lobeTypes = mi.getLobeTypes(sd);
        const bool hasNonDeltaLobes = (lobeTypes & (uint)LobeType::NonDelta) != 0;

        // Check if we should apply NEE.
        const bool applyNEE = hasNonDeltaLobes;

        // TODO: Support multiple shadow rays.
        path.setLightSampled(false, false);
        if (applyNEE)
        {
            LightSample ls = {};
            bool validSample = false;

            // Setup path vertex.
            PathVertex vertex = PathVertex(path.getVertexIndex(), sd.posW, sd.faceN, sd.frontFacing);

            // Determine if upper/lower hemispheres need to be sampled.
            bool sampleUpperHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaReflection) != 0);
            if (!kUseLightsInDielectricVolumes && path.isInsideDielectricVolume()) sampleUpperHemisphere = false;
            bool sampleLowerHemisphere = isCurveHit || isCurvePolyTubeHit || ((lobeTypes & (uint)LobeType::NonDeltaTransmission) != 0);

            // Sample a light.
            validSample = generateLightSample(vertex, sampleUpperHemisphere, sampleLowerHemisphere, path.sg, ls);
            path.setLightSampled(sampleUpperHemisphere, sampleLowerHemisphere);

            if (validSample)
            {
                // bool isPunctualLight = ls.pdf == 0.f;
                // if (ls.lightType == (uint)LightType::Analytic && ls.pdf == 0.f)
                //     ls.pdf = getAnalyticSelectionProbability(); // analytic light doesn't have a solid angle PDF.
                float3 Le = ls.Li * ls.pdf;

                // Apply MIS weight.
                float misWeight = 1.0f;
                float scatterPdf = 0.0f;
                if (ls.lightType != (uint)LightType::Analytic)
                {
                    scatterPdf = mi.evalPdf(sd, ls.dir, true);
                    misWeight = evalMIS(1, ls.pdf, 1, scatterPdf);
                }

                float3 weight = mi.eval(sd, ls.dir, path.sg);
                float3 Lr = weight * misWeight * ls.Li;
                if (any(Lr > 0.f))
                {
                    Ray ray = ls.getVisibilityRay();
                    // print("NEE Ray Origin", ray.origin);
                    // print("NEE Ray Dir", ray.dir);
                    // print("NEE Ray Dest", ray.origin + ray.dir * ray.tMax);

                    if (isCurvePolyTubeHit)
                    {
                        // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as light
                        // so there is no self-shadowing (transmission lobe of hair BSDF takes care of that).
                        if (dot(sd.frame.N, ray.dir) < 0.f)
                        {
                            ray.origin = ray.origin - sd.frame.N * sd.curveRadius * 2.1f;
                        }
                    }

                    if (traceVisibilityRay(ray))
                    {
                        bool selected = path.reservoir.addVertex(path.sg, 1, path.reconnection.earlyThroughput * path.thp * weight * ls.Li * misWeight);
                        // The path length from NEE is always greater than 1.
                        if (selected)
                        {
                            // Copy over any relevant reconnection data over.
                            path.reservoir.selectedReconnection = path.reconnection;

                            path.reservoir.selectedReconnection.pathLength = path.getVertexIndex() + 1;
                            path.reservoir.selectedReconnection.lightIsNEE = true;
                            path.reservoir.selectedReconnection.irradiance = Le;

                            if (path.getVertexIndex() == 1)
                                path.reservoir.selectedReconnection.transmissionEvent = dot(sd.faceN, sd.V) * dot(sd.faceN, ls.dir) < 0;

                            path.reservoir.selectedReconnection.lightIsDistant = (ls.lightType == (uint)LightType::EnvMap);

                            // print("NEE", path.reservoir.selectedReconnection.pathLength);
                            // print("MIS Weight", misWeight);
                            // print("Early Throughput", path.reconnection.earlyThroughput);
                            // print("Path Throughput", path.thp);
                            // print("Light NEE BSDF", weight);
                            // print("Light NEE PDF", ls.pdf);
                            // print("Light Emission", Le);
                            // print("Path Contribution", path.reconnection.earlyThroughput * path.thp * weight * ls.Li * misWeight);

                            if (path.getVertexIndex() == 1) // The selected light sample is the second vertex.
                            {
                                // Ignore other light types besides environment maps and emissive triangles for now.
                                if (ls.lightType == (uint)LightType::EnvMap /*|| (ls.lightType == (uint)LightType::Analytic && ls.distance == kMaxLightDistance)*/)
                                {
                                    path.reservoir.selectedReconnection.secondHit.data.x = 0;
                                    path.reservoir.selectedReconnection.secondHit.data.yzw = asuint(ls.dir);
                                    path.reservoir.selectedReconnection.secondaryPathJacobian = ls.pdf;
                                    path.reservoir.selectedReconnection.lightPdf = ls.pdf; // No need to convert to area measure because it's a distant light.
                                }
                                // else if (ls.lightType == (uint)LightType::Analytic)
                                // {
                                //     path.reconnection.secondLightReconnection = ls.origin + ls.dir * ls.distance;
                                //     if (isPunctualLight) // point
                                //     {
                                //         path.reconnection.irradiance /= ls.geometryTerm; // ls.geometry term is 1/d^2
                                //         path.reconnection.secondHit.data.x = 1;           // use this as type identifier
                                //     }
                                //     else // area, sphere
                                //     {
                                //         path.reconnection.secondHit.data.x = 2;
                                //         path.reconnection.secondHit.data.yzw = asuint(ls.normal);
                                //         path.reconnection.secondaryPathJacobian *= ls.geometryTerm;
                                //         path.reconnection.secondLightPdf *= ls.geometryTerm; // stored in area measure
                                //     }
                                // }
                                if (ls.lightType == (uint)LightType::Emissive)
                                {
                                    TriangleHit neeTriangleHit;
                                    uint2 meshPrimID = gScene.lightCollection.getSceneMeshPrimID(ls.triangleId);
                                    GeometryInstanceID neeGeoInstanceId;
                                    neeGeoInstanceId.index = meshPrimID.x;
                                    neeTriangleHit.instanceID = neeGeoInstanceId;
                                    neeTriangleHit.primitiveIndex = meshPrimID.y;
                                    neeTriangleHit.barycentrics = ls.barycentrics;
                                    HitInfo neeHitInfo = HitInfo(neeTriangleHit);

                                    path.reservoir.selectedReconnection.secondHit = neeHitInfo;

                                    path.reservoir.selectedReconnection.secondaryPathJacobian = ls.pdf * ls.geometryTerm;
                                    path.reservoir.selectedReconnection.lightPdf = ls.pdf * ls.geometryTerm; // Convert to area measure
                                }
                            }
                            else if (path.getVertexIndex() == 2) // The selected light sample is the third vertex.
                            {
                                // We are currently at the second vertex, and the selected light sample is the next direction.
                                path.reservoir.selectedReconnection.secondWo = ls.dir;

                                if (ls.lightType == (uint)LightType::Emissive || ls.lightType == (uint)LightType::EnvMap)
                                {
                                    // Keep in solid angle measure - NEE always has same probability of selecting the light at the reconnection vertex (does not depend on w_i),
                                    // so the Jacobian for p(y_{k + 1}) cancels out with each other.
                                    path.reservoir.selectedReconnection.lightPdf = ls.pdf;
                                }
                            }
                            else
                            {
                                path.reservoir.selectedReconnection.irradiance = path.thp * Lr;
                            }
                        }
                    }
                }
            }
        }

        // Russian roulette to terminate paths early.
        if (kUseRussianRoulette)
        {
            if (terminatePathByRussianRoulette(path, sampleNext1D(path.sg))) return sd.posW;
        }

        // Generate the next path segment or terminate.
        bool valid = generateScatterRay(sd, mi, path);

        // Check if this is the last path vertex.
        const bool isLastVertex = hasFinishedSurfaceBounces(path);

        // Terminate if this is the last path vertex and light sampling already completely sampled incident radiance.
        if (isLastVertex && path.isLightSamplable()) valid = false;

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }

        return sd.posW;
    }

    /** Handle the case when a scatter ray hits a surface, but don't try any NEE.
        After handling the hit, try.
        \param[in,out] path The path state.
        \param[in] isDirection If the provided second hit is a direction or a position.
        \param[in] secondHit Second hit, which can either be a direction or a position.
        \return The actual position of the hit.
    */
    float3 handleReconnectionHit(inout PathState path, const ReconnectionData reconnectionData, out float dstJacobian)
    {
        // Upon hit:
        // - Load vertex/material data
        // - Compute MIS weight if path.getVertexIndex() > 1 and emissive hit
        // - Add emitted radiance
        // - Generate next scatter ray based on second hit

        const bool isPrimaryHit = path.getVertexIndex() == 1;
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        const bool isCurveHit = kUseCurves && (hitType == HitType::Curve);

        // Load shading data. This is a long latency operation.
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);

        const bool isHairMaterial = kUseHairMaterial && (sd.mtl.getMaterialType() == MaterialType::Hair);
        // TODO: Decouple geometry from the material.
        const bool isCurvePolyTubeHit = isTriangleHit && isHairMaterial;

        // Account for volume absorption.
        if (!path.interiorList.isEmpty())
        {
            const uint materialID = path.interiorList.getTopMaterialID();
            let material = gScene.materials.getMaterial(materialID);
            let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
            path.thp *= HomogeneousVolumeSampler::evalTransmittance(vp, length(sd.posW - path.origin));
        }

        // Reject false hits in nested dielectrics.
        if (!handleNestedDielectrics(sd, path)) return sd.posW;

        // Create BSDF instance and query its properties.
        let hints = getMaterialInstanceHints(path.hit, isPrimaryHit);
        let lod = createTextureSampler(path, isPrimaryHit, isTriangleHit);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        // Disable specular lobes if caustics are disabled and path already contains a diffuse vertex.
        bool isSpecular = bsdfProperties.roughness <= params.specularRoughnessThreshold;
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && isSpecular)
        {
            sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
        }

        // Compute origin for rays traced from this path vertex.
        if (isCurveHit)
        {
            // For curves, we set the new origin at the sphere center.
            path.origin = sd.posW - sd.frame.N * sd.curveRadius;
        }
        else if (isCurvePolyTubeHit)
        {
            // For curves tessellated into poly-tubes, we offset the new origin away from the curve center.
            path.origin = sd.posW + sd.frame.N * sd.curveRadius * 0.1f;
        }
        else
        {
            path.origin = sd.computeRayOrigin();
        }

        if (reconnectionData.transmissionEvent)
        {
            if (isCurveHit)
            {
                // No need to offset the origin in this case.
            }
            else if (isCurvePolyTubeHit)
            {
                // For curves tessellated into poly-tubes, we make sure that the origin is on the same side as a scatter ray direction
                // so there is no self-intersection.
                path.origin = sd.posW - sd.frame.N * sd.curveRadius * 2.1f;
            }
            else
            {
                // Compute ray origin for next ray segment.
                path.origin = sd.computeRayOrigin(false);

                // Update interior list and inside volume flag.
                if (!sd.mtl.isThinSurface())
                {
                    uint nestedPriority = sd.mtl.getNestedPriority();
                    path.interiorList.handleIntersection(sd.materialID, nestedPriority, sd.frontFacing);
                    path.setInsideDielectricVolume(!path.interiorList.isEmpty());
                }
            }
        }

        bool secondHitIsDistant = (reconnectionData.pathLength == 2) && reconnectionData.lightIsDistant;

        dstJacobian = 1.0f;
        float3 nextDir = float3(0.0f, 0.0f, 0.0f);
        ShadingData secondSd = {};
        float rayLength = kRayTMax;

        if (!secondHitIsDistant)
        {
            secondSd = loadShadingData(reconnectionData.secondHit, path.origin, float3(0.0f, 0.0f, 0.0f));

            // Generate the next path segment to reconnect with the second hit.
            float3 shiftedOffset = secondSd.posW - sd.posW;
            float shiftedDist2 = dot(shiftedOffset, shiftedOffset);

            float shiftedCosine = abs(dot(secondSd.faceN, -normalize(shiftedOffset)));
            dstJacobian = shiftedCosine / shiftedDist2;

            // The ray is actually being shot from the path origin though,
            // so we should account for that.
            float3 adjustedOffset = secondSd.posW - path.origin;
            nextDir = normalize(adjustedOffset);
            rayLength = length(adjustedOffset) * 0.999f;

            // print("Second Hit Normal", secondSd.faceN);
            // print("Second Hit Position", secondSd.posW);
            // print("Second Hit Incoming Dir", -nextDir);
            // print("Shifted Cosine", shiftedCosine);
            // print("Shifted Distance", length(shiftedOffset));
            // print("Geometry Factor", dstJacobian);
        }
        else
        {
            nextDir = asfloat(reconnectionData.secondHit.data.yzw);
        }

        // print("Reconnection Hit", sd.posW);

        sd.mtl.setLobeMask(1 << reconnectionData.firstBSDFComponentType);
        float3 lobeScatterBsdf = mi.eval(sd, nextDir, path.sg);
        float lobeScatterPdf = mi.evalPdf(sd, nextDir, true);
        // print("Lobe Scatter BSDF", lobeScatterBsdf);
        // print("Lobe Scatter PDF", lobeScatterPdf);

        // We also need to adjust the Jacobian by the probability of sampling a given direction.
        // Here we use the total PDF when computing the MIS weights.
        sd.mtl.setLobeMask(0);
        float3 totalScatterBsdf = mi.eval(sd, nextDir, path.sg);
        float totalScatterPdf = mi.evalPdf(sd, nextDir, true);
        // print("Total Scatter BSDF", totalScatterBsdf);
        // print("Total Scatter PDF", totalScatterPdf);

        path.dir = nextDir;
        path.pdf = lobeScatterPdf;

        // TODO: this could probably be cleaned up quite a bit... conditionals are just being used right now for clarity.
        if (reconnectionData.pathLength == 2)
        {
            // Convert the light PDF back to the relevant solid angle measure.
            // print("Area Measure Light PDF", reconnectionData.lightPdf);
            float lightPdf = reconnectionData.lightPdf / dstJacobian;
            // print("Solid Angle Measure Light PDF", lightPdf);
            float misWeight = 1.0f;
            if (reconnectionData.lightIsNEE)
            {
                if (!any(totalScatterBsdf > 0.0f) || totalScatterPdf == 0.0f)
                {
                    path.terminate();
                    return sd.posW;
                }

                dstJacobian *= lightPdf;
                misWeight = evalMIS(1, lightPdf, 1, totalScatterPdf);
                path.thp *= totalScatterBsdf / lightPdf;

                // print("Dst Jacobian", dstJacobian);
                // print("MIS Weight", misWeight);
                // print("Light PDF", lightPdf);
                // print("Scatter BSDF", totalScatterBsdf);
            }
            else
            {
                if (!any(lobeScatterBsdf > 0.0f) || lobeScatterPdf == 0.0f)
                {
                    path.terminate();
                    return sd.posW;
                }

                dstJacobian *= lobeScatterPdf;
                misWeight = evalMIS(1, totalScatterPdf, 1, lightPdf);
                path.thp *= lobeScatterBsdf / lobeScatterPdf;
            }

            path.thp *= misWeight;
        }
        else
        {
            if (!any(lobeScatterBsdf > 0.0f) || lobeScatterPdf == 0.0f)
            {
                path.terminate();
                return sd.posW;
            }

            // print("Lobe Scatter BSDF", lobeScatterBsdf);
            // print("Lobe Scatter PDF", lobeScatterPdf);

            dstJacobian *= lobeScatterPdf;
            path.thp *= lobeScatterBsdf / lobeScatterPdf;
        }

        bool valid = any(path.thp > 0.f);

        // Terminate caustics paths.
        if (kDisableCaustics && path.getBounces(BounceType::Diffuse) > 0 && path.isSpecular()) valid = false;

        if (!valid)
        {
            path.terminate();
        }
        else
        {
            // If the ray is valid, we shoot a visibility ray to the second vertex (be careful about unintended intersections).
            Ray reconnectionRay = Ray(path.origin, path.dir, 0, rayLength);

            // If the visibility ray fails... the shift fails.
            if (!traceVisibilityRay(reconnectionRay))
            {
                path.terminate();
            }
        }

        return sd.posW;
    }

    float3 handleReconnectionPrimaryLight(inout PathState path)
    {
        const HitType hitType = path.hit.getType();
        const bool isTriangleHit = hitType == HitType::Triangle;
        let hints = getMaterialInstanceHints(path.hit, true);
        let lod = createTextureSampler(path, true, isTriangleHit);
        ShadingData sd = loadShadingData(path.hit, path.origin, path.dir);
        const IMaterialInstance mi = gScene.materials.getMaterialInstance(sd, lod, hints);
        BSDFProperties bsdfProperties = mi.getProperties(sd);

        return bsdfProperties.emission;
    }

    float3 handleReconnectionEnvMap(inout PathState path)
    {
        // Add env radiance.
        return (kUseEnvLight) ? envMapSampler.eval(path.dir, 0) : float3(0.0f, 0.0f, 0.0f);
    }

    /** Trace a visibility ray.
        \param[in] ray The visibility ray parameters
        \return If the ray is visible or not.
    */
    bool traceVisibilityRay(const Ray ray)
    {
        logTraceRay(PixelStatsRayType::Visibility);

        VisibilityQuery vq;
        return vq.traceVisibilityRay(ray);
    }

    /** Handle the case when a scatter ray misses the scene.
        \param[in,out] path The path state.
    */
    void handleMiss(inout PathState path)
    {
        // Upon miss:
        // - Compute MIS weight if previous path vertex sampled a light
        // - Evaluate environment map
        // - Write guiding data
        // - Terminate the path
        setupPathLogging(path);

        // Geometry factor is 1 because of an infinite distance.
        float prevScatterPdf = path.pdf;
        float geometryFactor = 1;

        // We can update common reconnection data at certain points.
        if (path.getVertexIndex() == 2)
        {
            path.reconnection.secondaryPathJacobian = prevScatterPdf * geometryFactor;
        }
        else if (path.getVertexIndex() == 3)
        {
            path.reconnection.secondaryPathJacobian *= prevScatterPdf;
        }

        // Check if the scatter event is samplable by the light sampling technique.
        const bool isLightSamplable = path.isLightSamplable();

        // Add env radiance.
        bool computeEnv = kUseEnvLight && (!path.isLightSampled() || !isLightSamplable);

        float3 emitterRadiance = 0.f;
        if (computeEnv)
        {
            logPathVertex();

            float misWeight = 1.f;
            float lightPdf = 0.0f;
            if (path.isLightSampled() && isLightSamplable)
            {
                // If NEE and MIS are enabled, and we've already sampled the env map,
                // then we need to evaluate the MIS weight here to account for the remaining contribution.

                // Evaluate PDF, had it been generated with light sampling.
                lightPdf = getEnvMapSelectionProbability() * envMapSampler.evalPdf(path.dir);

                // Compute MIS weight by combining this with BSDF sampling.
                // Note we can assume path.pdf > 0.f since we shouldn't have got here otherwise.
                misWeight = evalMIS(1, path.totalPdf, 1, lightPdf);
            }

            float3 Le = envMapSampler.eval(path.dir);
            emitterRadiance = misWeight * Le;

            bool selected = path.reservoir.addVertex(path.sg, 1, path.reconnection.earlyThroughput * path.thp * Le * misWeight);
            if (selected)
            {
                // Copy over any relevant reconnection data over.
                path.reservoir.selectedReconnection = path.reconnection;

                path.reservoir.selectedReconnection.pathLength = path.getVertexIndex();
                path.reservoir.selectedReconnection.lightIsNEE = false;
                path.reservoir.selectedReconnection.lightIsDistant = true;
                path.reservoir.selectedReconnection.lightPdf = lightPdf; // Distant light... so it should be position independent.

                // print("MISS BSDF", path.reservoir.selectedReconnection.pathLength);
                // print("MIS Weight", misWeight);
                // print("Early Throughput", path.reconnection.earlyThroughput);
                // print("Path Throughput", path.thp);
                // print("Light BSDF Emission", Le);
                // print("Path Contribution", path.reconnection.earlyThroughput * path.thp * Le * misWeight);

                if (path.getVertexIndex() == 2)
                {
                    path.reservoir.selectedReconnection.secondHit.data.x = 0;
                    path.reservoir.selectedReconnection.secondHit.data.yzw = asuint(path.dir);
                }

                // If the light is the second or third vertex,
                // We store the light PDF to re-calculate the MIS weights during reconnection.
                // We also store the incoming light directly because the throughput is re-calculated.
                if (path.getVertexIndex() == 2 || path.getVertexIndex() == 3)
                {
                    path.reservoir.selectedReconnection.irradiance = Le;
                }
                else
                {
                    path.reservoir.selectedReconnection.irradiance = path.thp * emitterRadiance;
                }
            }
        }

        path.terminate();
    }

    /** Handle the case when a primary ray misses the scene.
        \param[in,out] path The path state.
    */
    void handlePrimaryMiss(inout PathState path)
    {
        // Upon miss:
        // - Evaluate environment map
        // - Terminate the path
        setupPathLogging(path);

        // Add env radiance.
        if (kUseEnvLight)
        {
            logPathVertex();

            float3 Le = envMapSampler.eval(path.dir, 0);

            bool selected = path.reservoir.addVertex(path.sg, 1, Le);
            if (selected)
            {
                // A primary miss is never reconnectable.
                // Copy over any relevant reconnection data over.
                path.reservoir.selectedReconnection = path.reconnection;
                path.reservoir.selectedReconnection.lightIsDistant = true;

                Camera camera = gScene.camera;
                gCameraManager.orientCamera(camera, params.frameCount, path.time);
                float3 camDir = normalize(camera.data.cameraW);

                float cosSensor = dot(camDir, path.dir);
                path.reservoir.selectedReconnection.subPixelJacobian = 1.0f / abs(pow(cosSensor, 3));

                float focalPlaneZ = length(camera.data.cameraW);
                float d0 = focalPlaneZ / dot(camDir, path.dir);
                path.reservoir.selectedReconnection.lensVertexJacobian = (d0 * d0) / abs(cosSensor);

                path.reservoir.selectedReconnection.firstWi = -path.dir;
                path.reservoir.selectedReconnection.pathLength = 1;
                path.reservoir.selectedReconnection.irradiance = Le;
            }
        }

        path.terminate();
    }
};
