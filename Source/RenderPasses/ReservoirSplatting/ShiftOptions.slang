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
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

enum class TemporalReuse : uint32_t
{
    GatherOnly = 0,    ///< Gather reuse only.
    ScatterOnly = 1,   ///< Scatter reuse only.
    ScatterBackup = 2, ///< Scatter + backup.
    MultiScatter = 3,  ///< Scatter into multiple time partitions.
};

FALCOR_ENUM_INFO(
    TemporalReuse,
    {
        { TemporalReuse::GatherOnly, "GatherOnly" },
        { TemporalReuse::ScatterOnly, "ScatterOnly" },
        { TemporalReuse::ScatterBackup, "ScatterBackup" },
        { TemporalReuse::MultiScatter, "MultiScatter" },
    }
);
FALCOR_ENUM_REGISTER(TemporalReuse);

enum class ScatterBackupMIS : uint32_t
{
    Balance = 0,    ///< Generalized balance heuristic.
    Pairwise = 1,   ///< Pairwise. 
};

FALCOR_ENUM_INFO(
    ScatterBackupMIS,
    {
        { ScatterBackupMIS::Balance, "Balance" },
        { ScatterBackupMIS::Pairwise, "Pairwise" },
    }
);
FALCOR_ENUM_REGISTER(ScatterBackupMIS);

enum class GatherMechanism : uint32_t
{
    Fast,
    Clamped,
    Robust,
};

FALCOR_ENUM_INFO(
    GatherMechanism,
    {
        { GatherMechanism::Fast, "Fast" },
        { GatherMechanism::Clamped, "Clamped" },
        { GatherMechanism::Robust, "Robust" },
    }
);
FALCOR_ENUM_REGISTER(GatherMechanism);

END_NAMESPACE_FALCOR
