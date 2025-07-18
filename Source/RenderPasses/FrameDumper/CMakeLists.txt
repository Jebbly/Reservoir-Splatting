add_plugin(FrameDumper)

target_sources(FrameDumper PRIVATE
    FrameDumper.cpp
    FrameDumper.h
)

target_source_group(FrameDumper "RenderPasses")
