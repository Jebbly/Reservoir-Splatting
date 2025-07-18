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
#include "RaytracingCommon.slang"

import Utils.Color.ColorHelpers;
import Utils.Debug.PixelDebug;
import Utils.Sampling.SampleGenerator;

import LoadShadingData;
import PathTracer;
import ReconnectionData;

ParameterBlock<PathTracer> gPathTracer;

float computePrimaryHitCircleOfConfusion(float3 x1)
{
    Camera camera = gScene.camera;
    float lensRadius = camera.data.apertureRadius;

    // Compute via similar triangles;
    // Lens radius / lens distance to x1 == Film radius / film distance to x1. 
    float3 camDir = normalize(camera.data.cameraW);
    float camZ = dot(x1 - camera.data.posW, camDir);
    float filmZ = camZ - length(camera.data.cameraW);
    float filmRadius = abs(lensRadius / camZ * filmZ);

    // Normalize the film radius according to how far an adjacent pixel is, in pixels.
    float normalizedFilm = length(camera.data.cameraU / gPathTracer.params.frameDim.x + camera.data.cameraV / gPathTracer.params.frameDim.y);

    return filmRadius / normalizedFilm;
}

float computeEnvMapCircleOfConfusion()
{
    Camera camera = gScene.camera;
    float lensRadius = camera.data.apertureRadius;

    // With env maps, the fiml radius is exactly equal to the lens radius.
    float filmRadius = lensRadius;

    // Normalize the film radius according to how far an adjacent pixel is, in pixels.
    float normalizedFilm = length(camera.data.cameraU / gPathTracer.params.frameDim.x + camera.data.cameraV / gPathTracer.params.frameDim.y);

    return filmRadius / normalizedFilm;
}

float2 computeDepthOfFieldGatherShiftProbabilities(float r)
{
    float gamma = 0.2f + 6.2f / (r - 5.6f);
    gamma = (r <= 0.2f) ? 1.0f : clamp(gamma, 0.2f, 1.0f);
    return float2(gamma, 1.0f - gamma);
}

/**
 * Transform a timestamp and subpixel into a shifted sample (with a reconnection shift).
 * @param[in] sg Sample generator.
 * @param[inout] reconnectionData The reconnection data to use.
 * @param[in] time Timestamp of the shifted sample.
 * @param[in] fractionalPixel Fractional pixel of the shifted sample.
 * @return The radiance and Jacobians after the shift (in the relevant domain).
 */
ShiftedPathData gatherLensVertexCopyShift(SampleGenerator sg, inout ReconnectionData reconnectionData, const float time, const float2 fractionalPixel, const float2 lensSample)
{
    ShiftedPathData shiftedPath;

    // If the shifted path is out of bounds, return immediately.
    if (!gPathTracer.params.validateFloatingPixelBounds(fractionalPixel))
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    const bool hitEnvMap = (reconnectionData.pathLength == 1 && reconnectionData.lightIsDistant);

    // Trace the primary ray at a specific time, subpixel, and lens.
    PathState path = {};
    gPathTracer.generatePathWithParameters(path, asuint(sampleNext1D(sg)), time, fractionalPixel, lensSample);
    gPathTracer.nextHit(path);

    // Although the gather shift doesn't use the primary hit Jacobian, we still compute it here because it may be used in the future.
    Camera camera = gScene.camera;
    gCameraManager.orientCamera(camera, gPathTracer.params.frameCount, time);
    path.time = time;

    // Get the origin offset from the lens sample, which is always (0, 0) for a pinhole.
    float2 lensLocal = camera.data.apertureRadius * lensSample;
    float3 lensWorld = lensLocal.x * normalize(camera.data.cameraU) + lensLocal.y * normalize(camera.data.cameraV);
    path.origin = camera.data.posW + lensWorld;

    float3 camDir = normalize(camera.data.cameraW);

    shiftedPath.primaryHit = path.hit;
    shiftedPath.firstRayDir = path.dir;
    shiftedPath.fractionalPixel = fractionalPixel;
    shiftedPath.lensSample = lensSample;

    if (path.isHit())
    {
        // If we expected an environment map but hit something instead, the shift fails.
        if (hitEnvMap)
        {
            shiftedPath = ShiftedPathData();
            return shiftedPath;
        }

        ShadingData firstSd = loadShadingData(path.hit, path.origin, path.dir);
        float3 x1 = firstSd.posW;

        float dist = length(x1 - path.origin);
        float distSquared = dist * dist;
        float cosNormal = dot(-path.dir, firstSd.faceN);
        float cosSensor = dot(camDir, path.dir);
        shiftedPath.subPixelJacobian = abs(cosNormal / distSquared) / abs(pow(cosSensor, 3));

        float camZ = dot(x1 - camera.data.posW, camDir);
        float d0 = (length(camera.data.cameraW) / abs(camZ) * dist);
        float d1 = dist - d0;
        shiftedPath.lensVertexJacobian = (d0 * d0) / (d1 * d1) * abs(cosNormal) / abs(cosSensor);
    }
    else
    {
        // If we expected a hit but missed instead, the shift fails.
        if (!hitEnvMap)
        {
            shiftedPath = ShiftedPathData();
            return shiftedPath;
        }

        shiftedPath.radiance = gPathTracer.handleReconnectionEnvMap(path);

        float cosSensor = dot(camDir, path.dir);
        shiftedPath.subPixelJacobian = 1.0f / abs(pow(cosSensor, 3));

        float focalPlaneZ = length(camera.data.cameraW);
        float d0 = focalPlaneZ / dot(camDir, path.dir);
        shiftedPath.lensVertexJacobian = (d0 * d0) / abs(cosSensor);
        
        return shiftedPath;
    }

    // If the previous reservoir stores a primary light hit, we should now evaluate the emission at this primary hit.
    if (reconnectionData.pathLength == 1)
    {
        shiftedPath.radiance = gPathTracer.handleReconnectionPrimaryLight(path);
        return shiftedPath;
    }

    // Next compute the remaining path reconnection.
    shiftedPath.radiance = pathReconnectionShift(reconnectionData, path, shiftedPath.secondaryPathJacobian);

    return shiftedPath;
}

// Area ReSTIR's advanced DoF shift, which involves identifying a new lens vertex using the subpixel and primary hit.
ShiftedPathData gatherPrimaryHitReconnectionShift(SampleGenerator sg, inout ReconnectionData reconnectionData, const float time, const float2 fractionalPixel, const HitInfo primaryHit)
{
    ShiftedPathData shiftedPath;

    const bool hitEnvMap = (reconnectionData.pathLength == 1 && reconnectionData.lightIsDistant);

    PathState path = {};
    path.setActive();
    path.thp = float3(1.f);

    // Define the camera at a specific time.
    Camera camera = gScene.camera;
    gCameraManager.orientCamera(camera, gPathTracer.params.frameCount, time);
    path.time = time;

    // Find where the fractional pixel corresponds to the film (in world space).
    float2 p = fractionalPixel / gPathTracer.params.frameDim;
    float2 ndc = float2(2, -2) * p + float2(-1, 1);
    float3 filmWorld = camera.data.posW + (ndc.x * camera.data.cameraU + ndc.y * camera.data.cameraV + camera.data.cameraW);

    float3 x1, n;
    if (hitEnvMap)
    {
        path.dir = -reconnectionData.firstWi;
        x1 = filmWorld + path.dir;
        n = float3(0.0f, 0.0f, 0.0f);
    }
    else
    {
        ShadingData sd = loadShadingData(primaryHit, float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 0.0f));

        x1 = sd.posW;
        path.dir = normalize(x1 - filmWorld);
        n = sd.faceN;
    }

    // Compute where the ray would've started (along the lens plane).
    float3 camDir = normalize(camera.data.cameraW);
    float camZ = dot(x1 - camera.data.posW, camDir);
    float rayT = camZ / dot(path.dir, camDir);
    path.origin = x1 - rayT * path.dir;
    path.dir = hitEnvMap ? path.dir : normalize(x1 - path.origin); // For actual primary hits, recompute this to account for negatives when the primary hit is closer than the focal plane.

    // If the new ray would be outside the lens aperture, the shift fails.
    float3 lensOffsetWorld = path.origin - camera.data.posW;
    float2 lensLocalNormalized = float2(dot(lensOffsetWorld, normalize(camera.data.cameraU)), 
                                        dot(lensOffsetWorld, normalize(camera.data.cameraV)))
                                    / camera.data.apertureRadius;
    if (length(lensLocalNormalized) > 1.0f)
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    float dist = length(x1 - path.origin);
    float rayMin = hitEnvMap ? 0.001f : 0.001f * dist;
    float rayMax = hitEnvMap ? kRayTMax : 0.999f * dist;
    Ray visibilityRay = Ray(path.origin, path.dir, rayMin, rayMax);
    if (!gPathTracer.traceVisibilityRay(visibilityRay))
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    shiftedPath.firstRayDir = path.dir;
    shiftedPath.fractionalPixel = fractionalPixel;
    shiftedPath.lensSample = lensLocalNormalized;
    shiftedPath.primaryHit = primaryHit;

    // Create sample generator.
    path.sg = SampleGenerator(uint2(fractionalPixel), asuint(sampleNext1D(sg)));

    // Set the path hit.
    path.incrementVertexIndex();
    path.setHit(primaryHit);

    if (hitEnvMap)
    {
        shiftedPath.radiance = gPathTracer.handleReconnectionEnvMap(path);

        float cosSensor = dot(camDir, path.dir);
        shiftedPath.subPixelJacobian = 1.0f / abs(pow(cosSensor, 3));

        float focalPlaneZ = length(camera.data.cameraW);
        float d0 = focalPlaneZ / dot(camDir, path.dir);
        shiftedPath.lensVertexJacobian = (d0 * d0) / abs(dot(camDir, path.dir));

        return shiftedPath;
    }
    else
    {
        float dist = length(x1 - path.origin);
        float distSquared = dist * dist;
        float cosNormal = dot(path.dir, n);
        float cosSensor = dot(camDir, path.dir);
        shiftedPath.subPixelJacobian = abs(cosNormal / distSquared) / abs(pow(cosSensor, 3));

        float d0 = (length(camera.data.cameraW) / abs(camZ) * rayT);
        float d1 = rayT - d0;
        shiftedPath.lensVertexJacobian = (d0 * d0) / (d1 * d1) * (abs(cosNormal)) / (abs(cosSensor));
    }

    // If the previous reservoir stores a primary light hit, we should now evaluate the emission at this primary hit.
    if (reconnectionData.pathLength == 1)
    {
        shiftedPath.radiance = gPathTracer.handleReconnectionPrimaryLight(path);
        return shiftedPath;
    }

    // Next compute the remaining path reconnection.
    shiftedPath.radiance = pathReconnectionShift(reconnectionData, path, shiftedPath.secondaryPathJacobian);

    return shiftedPath;
}

// Some visibility checks are done during the scattering stage (e.g., from the previous to current frame, so don't check it again).
// Note that unlike the "time" parameter, the "pixel" parameter here is the domain we're shifting *from*. 
// It's only used when we have an environment map sample, and we need the fractional coord to reconstruct the ray direction.
ShiftedPathData scatterReprojectionShift(SampleGenerator sg, inout ReconnectionData reconnectionData, const float time, const HitInfo primaryHit, const float2 lensSample, bool skipVisibilityCheck = false)
{
    ShiftedPathData shiftedPath;

    const bool hitEnvMap = (reconnectionData.pathLength == 1) && reconnectionData.lightIsDistant;

    PathState path = {};
    path.setActive();
    path.thp = float3(1.f);

    // Orient the camera at the given time.
    Camera camera = gScene.camera;
    gCameraManager.orientCamera(camera, gPathTracer.params.frameCount, time);
    path.time = time;

    // Get the origin offset from the lens sample, which is always (0, 0) for a pinhole.
    float2 lensLocal = camera.data.apertureRadius * lensSample;
    float3 camU = normalize(camera.data.cameraU);
    float3 camV = normalize(camera.data.cameraV);
    float3 lensWorld = lensLocal.x * camU + lensLocal.y * camV;
    path.origin = camera.data.posW + lensWorld;

    float3 x1, n;
    if (hitEnvMap)
    {
        path.dir = -reconnectionData.firstWi;
    }
    else
    {
        ShadingData firstSd = loadShadingData(primaryHit, camera.data.posW, float3(0.0f, 0.0f, 0.0f));
        x1 = firstSd.posW;
        n = firstSd.faceN;

        path.dir = normalize(x1 - path.origin);
    }

    // Convert the new ray to camera space.
    float3 camDir = normalize(camera.data.cameraW);
    float3 camRay = float3(dot(camU, path.dir), dot(camV, path.dir), dot(camDir, path.dir));

    // If the new position is behind the camera, don't scatter.
    if (camRay.z <= 0.001f)
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    // Find the location on the physical film, and convert to NDC -> subpixel location.
    float2 film = lensLocal + length(camera.data.cameraW) * (camRay.xy / camRay.z);
    float2 ndc = film / float2(length(camera.data.cameraU), length(camera.data.cameraV));
    float2 newFractionalPixel = (float2(0.5f, -0.5f) * ndc + float2(0.5f, 0.5f)) * gPathTracer.params.frameDim;

    // If the new position doesn't lie in the camera frame, don't scatter.
    if (!gPathTracer.params.validateFloatingPixelBounds(newFractionalPixel))
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    // Check that the reprojection is actually visible.
    float dist = length(x1 - path.origin);
    float rayMax = hitEnvMap ? kRayTMax : 0.999f * dist;
    Ray visibilityRay = Ray(path.origin, path.dir, 0.001f, rayMax);
    if (!skipVisibilityCheck && !gPathTracer.traceVisibilityRay(visibilityRay))
    {
        shiftedPath = ShiftedPathData();
        return shiftedPath;
    }

    // At this point, the reprojection has succeeeded, so the next step is to compute the Jacobians and the remaining reconnection shift.
    shiftedPath.lensSample = lensSample;
    shiftedPath.firstRayDir = path.dir;
    shiftedPath.fractionalPixel = newFractionalPixel;
    shiftedPath.primaryHit = primaryHit;

    // Create sample generator.
    path.sg = SampleGenerator(uint2(newFractionalPixel), asuint(sampleNext1D(sg)));

    // Set the path hit.
    path.incrementVertexIndex();
    path.setHit(primaryHit);

    if (hitEnvMap)
    {
        shiftedPath.radiance = gPathTracer.handleReconnectionEnvMap(path);

        float cosSensor = dot(camDir, path.dir);
        shiftedPath.subPixelJacobian = 1.0f / abs(pow(cosSensor, 3));

        float focalPlaneZ = length(camera.data.cameraW);
        float d0 = focalPlaneZ / dot(camDir, path.dir);
        shiftedPath.lensVertexJacobian = (d0 * d0) / abs(cosSensor);

        return shiftedPath;
    }
    else
    {
        // Compute the Jacobian.
        float distSquared = dist * dist;
        float cosNormal = dot(-path.dir, n);
        float cosSensor = dot(normalize(camera.data.cameraW), path.dir);
        shiftedPath.subPixelJacobian = abs(cosNormal / distSquared) / abs(pow(cosSensor, 3));

        float camZ = dot(x1 - camera.data.posW, camDir);
        float d0 = (length(camera.data.cameraW) / abs(camZ) * dist);
        float d1 = dist - d0;
        shiftedPath.lensVertexJacobian = (d0 * d0) / (d1 * d1) * abs(cosNormal) / abs(cosSensor);
    }
    
    // If our temporal reservoir only had a hit length of 1 at this point, we must've hit a light source.
    if (reconnectionData.pathLength == 1)
    {
        shiftedPath.radiance = gPathTracer.handleReconnectionPrimaryLight(path);
        return shiftedPath;
    }

    // Next compute the remaining path reconnection.
    shiftedPath.radiance = pathReconnectionShift(reconnectionData, path, shiftedPath.secondaryPathJacobian);

    return shiftedPath;
}

/**
 * Transform a path with a primary hit into a fully shifted sample.
 * @param[inout] reconnectionData The reconnection data to use.
 * @param[inout] path Path state after the primary hit.
 * @param[out] dstJacobian The destination Jacobian of the shift.
 * @return The radiance after the shift (in the relevant domain).
 */
float3 pathReconnectionShift(inout ReconnectionData reconnectionData, inout PathState path, out float dstJacobian)
{
    // At this point, we have found a valid primary hit.
    // The reconnection path already provides a second hit (or direction),
    // so shoot a ray in that direction (with the correct throughput).
    dstJacobian = 1.0f;
    float3 y_1 = gPathTracer.handleReconnectionHit(path, reconnectionData, dstJacobian);

    // If the path is not valid, then the shift fails.
    if (!path.isActive())
    {
        return float3(0.0f, 0.0f, 0.0f);
    }

    ShadingData secondSd = loadShadingData(reconnectionData.secondHit, path.origin, path.dir);

    // There are three cases now: 
    // If the path is length 2, then we directly use the irradiance (MIS weights are already accounted for).
    // If the path length is 3, then we need to recompute the MIS weight as well.
    // If the path length is greater than 3, then we need to update the throughput before using the irradiance.
    let hints = getMaterialInstanceHints(path.hit, false);
    let lod = ExplicitLodTextureSampler(0);
    const IMaterialInstance mi = gScene.materials.getMaterialInstance(secondSd, lod, hints);

    secondSd.mtl.setLobeMask(1 << reconnectionData.secondBSDFComponentType);
    float3 lobeSpecificBsdf = mi.eval(secondSd, reconnectionData.secondWo, path.sg);
    float lobeSpecificPdf = mi.evalPdf(secondSd, reconnectionData.secondWo, true);

    secondSd.mtl.setLobeMask(0xF);
    float3 totalBsdf = mi.eval(secondSd, reconnectionData.secondWo, path.sg);
    float totalPdf = mi.evalPdf(secondSd, reconnectionData.secondWo, true);

    // TODO: this could probably also be placed in a better location.
    if (reconnectionData.pathLength > 2 && !any(totalBsdf > 0.0f))
    {
        return float3(0.0f, 0.0f, 0.0f);
    }

    float misWeight = 1.0f;
    if (reconnectionData.pathLength == 3)
    {
        bool valid = reconnectionData.lightIsNEE ? true : (lobeSpecificPdf > 0.0f);
        if (!valid)
        {
            return float3(0.0f, 0.0f, 0.0f);
        }

        // When the path length is only 3 and the light was sampled by NEE,
        // the reconnection data's lightPdf stores the solid angle measure instead of the area measure.
        misWeight = reconnectionData.lightIsNEE ? PathTracer::evalMIS(1, reconnectionData.lightPdf, 1, totalPdf) :
                                                      PathTracer::evalMIS(1, totalPdf, 1, reconnectionData.lightPdf);
        float3 lightConnectionThp = reconnectionData.lightIsNEE ? totalBsdf / reconnectionData.lightPdf : lobeSpecificBsdf / lobeSpecificPdf;

        // print("Lobe Specific BSDF", lobeSpecificBsdf);
        // print("Lobe Specific Pdf", lobeSpecificPdf);
        // print("Total BSDF", totalBsdf);
        // print("Light PDF", reconnectionData.lightPdf);
        // print("Light Connection Throughput", lightConnectionThp);

        path.thp *= lightConnectionThp;

        // If the light was selected by NEE, the src / dst Jacobians cancel out.
        // Otherwise, we must account for the difference in BSDF sampling.
        dstJacobian *= reconnectionData.lightIsNEE ? 1.0f : lobeSpecificPdf;
    }
    else if (reconnectionData.pathLength > 3)
    {
        if (!(lobeSpecificPdf > 0.0f))
        {
            return float3(0.0f, 0.0f, 0.0f);
        }
        path.thp *= lobeSpecificBsdf / lobeSpecificPdf;
        dstJacobian *= lobeSpecificPdf;

        // print("Lobe Specific Bsdf", lobeSpecificBsdf);
        // print("Lobe Specific Pdf", lobeSpecificPdf);
    }

    // print("MIS Weight", misWeight);
    // print("Final Path Throughput", path.thp);
    // print("Final Irradiance", reconnectionData.irradiance);
    // print("Dst Jacobian", dstJacobian);
    return misWeight * path.thp * reconnectionData.irradiance;
}
