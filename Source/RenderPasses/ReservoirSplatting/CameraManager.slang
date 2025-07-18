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

import Scene.Camera.Camera;
import Utils.Debug.PixelDebug;
import Utils.Math.Quaternion;

struct CameraManager {
    float artificialFrameTime; // Artificially defined time between frames.

    /// Used for free camera motion. The number of entries is set at shader compile time,
    /// so unfortunately, this cannot be changed to the
    static const uint kCameraKeyframes = NUM_CAMERA_SAMPLES;
    float3 cameraPositions[kCameraKeyframes];
    float3 cameraTargets[kCameraKeyframes];

    /**
     * Set the correct camera vectors according to the rotation matrix.
     * @param[in] prevForward Previous forward vector.
     * @param[in] currFoward Current forward vector.
     * @param[in] t Interpolation value.
     * @return The interpolated forward vector.
     *
     */
    float3 angularLerp(float3 prevForward, float3 currForward, float t)
    {
        // From https://www.gamedev.net/forums/topic/523136-slerping-two-vectors/523136/
        float3 axis = normalize(cross(currForward, prevForward));
        float angle = acos(dot(currForward, prevForward));

        if (angle < 0.001f) {
            // Both prevForward and currForward are basically the same thing here.
            return currForward;
        }
        float4 quat = rotate_angle_axis(angle * t, axis);

        return rotate_vector(currForward, quat);
    }

    /**
     * Set the correct camera vectors according to the forward vector.
     * @param[in,out] camera Camera to orient.
     * @param[in] forward Camera's normalized forward vector.
     *
     */
    void recalculateCameraVectors(inout Camera camera, float3 forward) {
        camera.data.cameraW = forward * camera.data.focalDistance;
        camera.data.cameraU = normalize(cross(camera.data.cameraW, camera.data.up));
        camera.data.cameraV = normalize(cross(camera.data.cameraU, camera.data.cameraW));

        float fovY = camera.data.focalLength == 0.0f ? 0.0f : 2.0f * atan(0.5f * camera.data.frameHeight / camera.data.focalLength);
        float ulen = camera.data.focalDistance * tan(fovY * 0.5f) * camera.data.aspectRatio;
        float vlen = camera.data.focalDistance * tan(fovY * 0.5f);

        camera.data.cameraU *= ulen;
        camera.data.cameraV *= vlen;
    }

    /**
     * Orients the camera according to some given timestamp.
     * @param[in,out] camera Camera to orient.
     * @param[in] frameCount Frame count to track current index of circular buffer.
     * @param[in] time Randomly sampled time.
     *
     */
    void orientCamera(inout Camera camera, uint frameCount, float time) {
        // First find which 2 frames to interpolate between.
        int currIndex = frameCount % kCameraKeyframes; // index in the circular buffer

        int frameIndex = (int)floor(time / artificialFrameTime);
        int firstCameraIndex = (currIndex - frameIndex - 1 + kCameraKeyframes) % kCameraKeyframes;
        int secondCameraIndex = (firstCameraIndex + 1 + kCameraKeyframes) % kCameraKeyframes;
        float t = (time - frameIndex * artificialFrameTime) / artificialFrameTime;

        float3 firstCameraPosition = cameraPositions[firstCameraIndex];
        float3 firstCameraTarget = cameraTargets[firstCameraIndex];
        float3 secondCameraPosition = cameraPositions[secondCameraIndex];
        float3 secondCameraTarget = cameraTargets[secondCameraIndex];

        camera.data.up = float3(0.0f, 1.0f, 0.0f);
        camera.data.posW = lerp(secondCameraPosition, firstCameraPosition, t);
        float3 firstForward = normalize(firstCameraTarget - firstCameraPosition);
        float3 secondForward = normalize(secondCameraTarget - secondCameraPosition);
        float3 interpolatedForward = angularLerp(firstForward, secondForward, t);
        recalculateCameraVectors(camera, interpolatedForward);

        return;
    }
};

ParameterBlock<CameraManager> gCameraManager;
