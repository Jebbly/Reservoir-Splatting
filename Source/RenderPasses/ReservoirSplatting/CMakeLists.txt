add_plugin(ReservoirSplatting)

target_sources(ReservoirSplatting PRIVATE
    ReservoirSplatting.cpp
    ReservoirSplatting.h
    # Common Utility
    CameraManager.slang
    LoadShadingData.slang
    PathTracer.slang
    Params.slang
    PathStates.slang
    ReflectTypes.cs.slang
    StaticParams.slang
    RaytracingCommon.slang
    # Shift Utility
    Reservoir.slang
    ReconnectionData.slang
    ShiftOptions.slang
    ShiftMapping.slang
    # Initial Candidates
    InitialCandidates.cs.slang
    # Gather Temporal Resampling
    GatherHelper.slang
    RobustReuseOptimization.rt.slang
    CollectTemporalSamples.cs.slang
    GatherTemporalResampling.rt.slang
    # Scatter Temporal Resampling
    ReprojectTemporalSamples.rt.slang
    SortReprojectedReservoirs.cs.slang
    ScatterTemporalResampling.rt.slang
    # Scatter + Backup Temporal Resampling
    ScatterBackupTemporalResampling.rt.slang
    # Multi-Scatter Temporal Resampling
    MultiReprojectTemporalSamples.rt.slang
    MultiSortReprojectedReservoirs.cs.slang
    MultiScatterTemporalResampling.rt.slang
    # Spatial Resampling
    SpatialResampling.rt.slang
    # Resolve ReSTIR
    ResolveReSTIR.cs.slang
    # Visualization Utility
    VisualizeForwardReprojection.cs.slang
)

target_copy_shaders(ReservoirSplatting RenderPasses/ReservoirSplatting)

target_source_group(ReservoirSplatting "RenderPasses")
