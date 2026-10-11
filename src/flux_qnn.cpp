// flux-qnn: run a QNN context binary (one float tensor in, one out) on the NPU - no Python, no ONNX Runtime.
//
// 2026-10-02. flux.py runs the VAE decoder / encoder through ONNX Runtime's QNN EP, which compiles the ONNX for the
// HTP once and saves the compiled graph as <name>_ctx_qnn.bin. That file is a plain QNN context binary: this program
// loads it with QnnHtp.dll alone (flux_qnn_graph.hpp on top of pcore_qnn_min.hpp, as the CCTV app's detector and
// upscaler do it) and executes. It is the proof - and the building block - for a picture app without Python:
// tools\qnn_cpp_check_1002.py compares its picture with ONNX Runtime's (identical pixels, 396 ms at 512 x 512).
//   flux-qnn <ctx_qnn.bin> <in.f32> <out.f32> [folder with QnnHtp.dll and QnnSystem.dll]
// in.f32 / out.f32: raw float32 in the tensor's own layout (the decoder: [1, 32, 2Hp, 2Wp] in, [1, 3, H, W] out).
// stdout: "DONE <ms>".

#include "flux_qnn_graph.hpp"

static int run(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: flux-qnn <ctx_qnn.bin> <in.f32> <out.f32> [folder with QnnHtp.dll]\n"); return 2; }
    const std::string bin = argv[1], in_path = argv[2], out_path = argv[3];
    if (argc > 4) SetDllDirectoryA(argv[4]);
    pcore_npu::QnnHtpSession S;                       // closed by its destructor, after the graph's context
    if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
    fluxqnn::Graph G;
    G.load(S, bin);
    const std::vector<uint8_t> raw = fluxqnn::read_file(in_path);
    FILE * f = nullptr;
    if (G.in.is_u8() && G.out.is_u8()) {              // 10-02: a quantised network (MiDaS depth, an upscaler): bytes in, bytes out
        if (raw.size() != G.in.count) throw std::runtime_error("the input file has " + std::to_string(raw.size()) + " bytes, the graph wants " + G.in.str());
        const std::vector<uint8_t> res = G.run_u8(raw.data());
        if (!(f = fopen(out_path.c_str(), "wb"))) throw std::runtime_error("cannot write " + out_path);
        fwrite(res.data(), 1, res.size(), f);
    } else {
        if (raw.size() != G.in.count * 4) throw std::runtime_error("the input file has " + std::to_string(raw.size() / 4) + " floats, the graph wants " + G.in.str());
        const std::vector<float> res = G.run((const float *) raw.data());
        if (!(f = fopen(out_path.c_str(), "wb"))) throw std::runtime_error("cannot write " + out_path);
        fwrite(res.data(), 4, res.size(), f);
    }
    fclose(f);
    fprintf(stderr, "[qnn] %s: in %s, out %s | load %.0f ms, execute %.0f ms, power profile %s\n", G.name.c_str(), G.in.str().c_str(), G.out.str().c_str(),
            G.load_ms, G.run_ms, S.perf_ok() ? "set" : "not set");
    printf("DONE %.0f\n", G.run_ms);
    fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[qnn] stopped: %s\n", e.what());
        return 1;
    }
}
