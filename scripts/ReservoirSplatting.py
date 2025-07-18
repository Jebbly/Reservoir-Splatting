from falcor import *

def render_graph_ReservoirSplatting():
    g = RenderGraph("ReservoirSplatting")

    VBufferParams = {
        'samplePattern': "Center",
        'sampleCount': 1,
        'useAlphaTest': True,
        'useDOF' : False
    }

    ReservoirSplattingParams = {
        'samplesPerPixel': 1,
        'enableTemporalResampling': True,
        'enableSpatialResampling': True,
        'temporalReuse': "ScatterOnly",
        'numTimePartitions': 2,
        'hybridMISOption': "Balance",
        'gatherOption': "Fast",
    }

    VBufferRT = createPass("VBufferRT", VBufferParams)
    g.addPass(VBufferRT, "VBufferRT")
    ReservoirSplatting = createPass("ReservoirSplatting", ReservoirSplattingParams)
    g.addPass(ReservoirSplatting, "ReservoirSplatting")
    AccumulatePass = createPass("AccumulatePass", {'enabled': False, 'precisionMode': 'Single'})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {'autoExposure': False, 'exposureCompensation': 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    FrameDumper = createPass("FrameDumper")
    g.addPass(FrameDumper, "FrameDumper")
    g.addEdge("VBufferRT.vbuffer", "ReservoirSplatting.vbuffer")
    # g.addEdge("VBufferRT.viewW", "PathTracer.viewW")
    g.addEdge("VBufferRT.mvec", "ReservoirSplatting.mvec")
    g.addEdge("ReservoirSplatting.color", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    # g.markOutput("ToneMapper.dst")
    g.addEdge("ToneMapper.dst", "FrameDumper.src")
    g.markOutput("FrameDumper.dst")
    g.markOutput("AccumulatePass.output")
    return g

ReservoirSplatting = render_graph_ReservoirSplatting()
try: m.addGraph(ReservoirSplatting)
except NameError: None
