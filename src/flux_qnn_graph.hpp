// flux_qnn_graph.hpp - one QNN context binary (one float tensor in, one out) as an object: load, run, gone.
// 2026-10-02. The session is pcore_qnn_min.hpp (QnnHtp.dll alone); graph name and tensors are read from the binary
// with QnnSystem.dll. Used by flux-qnn.exe (the stand-alone check) and flux-make.exe (the picture stage in-process).
// Float16 tensors are converted here; everything the caller sees is float32 in the tensor's own layout.
#pragma once
#include "pcore_qnn_min.hpp"
#include <QnnContext.h>
#include <QnnGraph.h>
#include <QnnTensor.h>
#include <System/QnnSystemContext.h>
#include <System/QnnSystemInterface.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace fluxqnn {

inline double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline std::vector<uint8_t> read_file(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + p);
    _fseeki64(f, 0, SEEK_END);
    const long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    std::vector<uint8_t> v((size_t) n);
    const size_t got = fread(v.data(), 1, v.size(), f);
    fclose(f);
    if (got != v.size()) throw std::runtime_error("short read " + p);
    return v;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t) ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t) (sign | 0x7C00u | (man ? 0x200u : 0));
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) sign;
        man |= 0x800000u;
        const uint32_t shift = (uint32_t) (14 - exp);
        uint32_t h = man >> shift;
        if ((man >> (shift - 1)) & 1u) h++;
        return (uint16_t) (sign | h);
    }
    uint32_t h = ((uint32_t) exp << 10) | (man >> 13);
    if (man & 0x1000u) h++;
    return (uint16_t) (sign | h);
}
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu, x;
    if (exp == 0) {
        if (man == 0) x = sign;
        else { exp = 1; while (!(man & 0x400u)) { man <<= 1; exp--; } man &= 0x3FFu; x = sign | ((exp + 112) << 23) | (man << 13); }
    } else if (exp == 31) x = sign | 0x7F800000u | (man << 13);
    else x = sign | ((exp + 112) << 23) | (man << 13);
    float f;
    memcpy(&f, &x, 4);
    return f;
}

struct Io {            // one graph tensor as the binary describes it
    Qnn_Tensor_t t{};
    std::vector<uint32_t> dims;
    Qnn_DataType_t dt = QNN_DATATYPE_FLOAT_32;
    size_t count = 1;
    std::string str() const {
        std::string r = "[";
        for (size_t i = 0; i < dims.size(); i++) r += (i ? "," : "") + std::to_string(dims[i]);
        return r + (dt == QNN_DATATYPE_FLOAT_16 ? "] f16" : dt == QNN_DATATYPE_FLOAT_32 ? "] f32" : is_u8() ? "] u8" : "] type " + std::to_string((int) dt));
    }
    bool is_float() const { return dt == QNN_DATATYPE_FLOAT_32 || dt == QNN_DATATYPE_FLOAT_16; }
    bool is_u8() const { return dt == QNN_DATATYPE_UFIXED_POINT_8 || dt == QNN_DATATYPE_UINT_8; }
};

// One loaded graph. The SESSION is the caller's and must outlive every Graph made on it.
class Graph {
public:
    Io in, out;
    std::string name;
    double load_ms = 0, run_ms = 0;

    Graph() = default;
    Graph(const Graph &) = delete;
    Graph & operator=(const Graph &) = delete;
    ~Graph() { if (ctx_ && iface_) iface_->QNN_INTERFACE_VER_NAME.contextFree(ctx_, nullptr); }

    void load(pcore_npu::QnnHtpSession & S, const std::string & bin) {
        const double t0 = now_ms();
        blob_ = read_file(bin);
        HMODULE hs = LoadLibraryA("QnnSystem.dll");
        if (!hs) throw std::runtime_error("cannot load QnnSystem.dll");
        using SGP = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *);
        auto sgp = (SGP) GetProcAddress(hs, "QnnSystemInterface_getProviders");
        const QnnSystemInterface_t ** sp = nullptr;
        uint32_t nsp = 0;
        if (!sgp || sgp(&sp, &nsp) != QNN_SUCCESS || !nsp) throw std::runtime_error("QnnSystemInterface_getProviders failed");
        auto SI = sp[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
        QnnSystemContext_Handle_t sc = nullptr;
        if (SI.systemContextCreate(&sc) != QNN_SUCCESS) throw std::runtime_error("systemContextCreate failed");
        const QnnSystemContext_BinaryInfo_t * bi = nullptr;
        Qnn_ContextBinarySize_t bsz = 0;
        if (SI.systemContextGetBinaryInfo(sc, blob_.data(), (uint64_t) blob_.size(), &bi, &bsz) != QNN_SUCCESS || !bi)
            throw std::runtime_error("not a QNN context binary QnnSystem can read: " + bin);
        uint32_t ng = 0;
        QnnSystemContext_GraphInfo_t * gs = nullptr;
        if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) { ng = bi->contextBinaryInfoV1.numGraphs; gs = bi->contextBinaryInfoV1.graphs; }
        else if (bi->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) { ng = bi->contextBinaryInfoV2.numGraphs; gs = bi->contextBinaryInfoV2.graphs; }
        else { ng = bi->contextBinaryInfoV3.numGraphs; gs = bi->contextBinaryInfoV3.graphs; }
        if (!ng || !gs) throw std::runtime_error("the binary has no graph: " + bin);
        const char * gn = nullptr;
        uint32_t ni = 0, no = 0;
        Qnn_Tensor_t * ins = nullptr, * outs = nullptr;
        auto & g = gs[0];
        if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) { gn = g.graphInfoV1.graphName; ni = g.graphInfoV1.numGraphInputs; ins = g.graphInfoV1.graphInputs; no = g.graphInfoV1.numGraphOutputs; outs = g.graphInfoV1.graphOutputs; }
        else if (g.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) { gn = g.graphInfoV2.graphName; ni = g.graphInfoV2.numGraphInputs; ins = g.graphInfoV2.graphInputs; no = g.graphInfoV2.numGraphOutputs; outs = g.graphInfoV2.graphOutputs; }
        else { gn = g.graphInfoV3.graphName; ni = g.graphInfoV3.numGraphInputs; ins = g.graphInfoV3.graphInputs; no = g.graphInfoV3.numGraphOutputs; outs = g.graphInfoV3.graphOutputs; }
        if (!gn || ni != 1 || no != 1) throw std::runtime_error("expected one input and one output, the graph has " + std::to_string(ni) + " and " + std::to_string(no));
        name = gn;
        in = io_of(ins[0]);
        out = io_of(outs[0]);
        SI.systemContextFree(sc);
        for (const Io * io : { &in, &out })
            if (!io->is_float() && !io->is_u8()) throw std::runtime_error("only float and 8-bit tensors are handled, got " + io->str());
        iface_ = S.iface();
        auto & F = iface_->QNN_INTERFACE_VER_NAME;
        if (F.contextCreateFromBinary(S.backend(), S.device(), nullptr, blob_.data(), (Qnn_ContextBinarySize_t) blob_.size(), &ctx_, nullptr) != QNN_SUCCESS || !ctx_)
            throw std::runtime_error("contextCreateFromBinary failed (a binary of another QNN runtime?): " + bin);
        if (F.graphRetrieve(ctx_, name.c_str(), &graph_) != QNN_SUCCESS || !graph_) throw std::runtime_error("graphRetrieve failed for " + name);
        std::vector<uint8_t>().swap(blob_);            // the context holds its own copy now
        load_ms = now_ms() - t0;
    }

    // 8-bit in, 8-bit out (a quantised picture network): the bytes as they are, in.count of them
    std::vector<uint8_t> run_u8(const uint8_t * x) {
        if (!in.is_u8() || !out.is_u8()) throw std::runtime_error("run_u8 on a graph that is not 8-bit: " + in.str() + " -> " + out.str());
        std::vector<uint8_t> obuf(out.count);
        bind(in, (void *) x, in.count, QNN_TENSOR_TYPE_APP_WRITE);
        bind(out, obuf.data(), obuf.size(), QNN_TENSOR_TYPE_APP_READ);
        const double t0 = now_ms();
        if (iface_->QNN_INTERFACE_VER_NAME.graphExecute(graph_, &in.t, 1, &out.t, 1, nullptr, nullptr) != QNN_SUCCESS) throw std::runtime_error("graphExecute failed");
        run_ms = now_ms() - t0;
        return obuf;
    }

    // x: in.count floats; returns out.count floats
    std::vector<float> run(const float * x) {
        if (!in.is_float() || !out.is_float()) throw std::runtime_error("run on a graph that is not float: " + in.str() + " -> " + out.str());
        std::vector<uint8_t> ibuf(in.count * (in.dt == QNN_DATATYPE_FLOAT_16 ? 2 : 4)), obuf(out.count * (out.dt == QNN_DATATYPE_FLOAT_16 ? 2 : 4));
        if (in.dt == QNN_DATATYPE_FLOAT_16) { uint16_t * d = (uint16_t *) ibuf.data(); for (size_t i = 0; i < in.count; i++) d[i] = f32_to_f16(x[i]); }
        else memcpy(ibuf.data(), x, in.count * 4);
        bind(in, ibuf.data(), ibuf.size(), QNN_TENSOR_TYPE_APP_WRITE);
        bind(out, obuf.data(), obuf.size(), QNN_TENSOR_TYPE_APP_READ);
        const double t0 = now_ms();
        if (iface_->QNN_INTERFACE_VER_NAME.graphExecute(graph_, &in.t, 1, &out.t, 1, nullptr, nullptr) != QNN_SUCCESS) throw std::runtime_error("graphExecute failed");
        run_ms = now_ms() - t0;
        std::vector<float> res(out.count);
        if (out.dt == QNN_DATATYPE_FLOAT_16) { const uint16_t * s = (const uint16_t *) obuf.data(); for (size_t i = 0; i < out.count; i++) res[i] = f16_to_f32(s[i]); }
        else memcpy(res.data(), obuf.data(), out.count * 4);
        return res;
    }

private:
    static Io io_of(const Qnn_Tensor_t & src) {
        Io r;
        r.t = src;
        const uint32_t rank = src.version == QNN_TENSOR_VERSION_1 ? src.v1.rank : src.v2.rank;
        const uint32_t * d = src.version == QNN_TENSOR_VERSION_1 ? src.v1.dimensions : src.v2.dimensions;
        r.dt = src.version == QNN_TENSOR_VERSION_1 ? src.v1.dataType : src.v2.dataType;
        r.dims.assign(d, d + rank);
        for (uint32_t v : r.dims) r.count *= v;
        return r;
    }
    static void bind(Io & io, void * data, size_t bytes, Qnn_TensorType_t ty) {
        if (io.t.version == QNN_TENSOR_VERSION_1) {
            io.t.v1.type = ty; io.t.v1.memType = QNN_TENSORMEMTYPE_RAW; io.t.v1.dimensions = io.dims.data();
            io.t.v1.clientBuf.data = data; io.t.v1.clientBuf.dataSize = (uint32_t) bytes;
        } else {
            io.t.v2.type = ty; io.t.v2.memType = QNN_TENSORMEMTYPE_RAW; io.t.v2.dimensions = io.dims.data();
            io.t.v2.clientBuf.data = data; io.t.v2.clientBuf.dataSize = (uint32_t) bytes;
        }
    }
    const QnnInterface_t * iface_ = nullptr;
    Qnn_ContextHandle_t ctx_ = nullptr;
    Qnn_GraphHandle_t graph_ = nullptr;
    std::vector<uint8_t> blob_;
};

}  // namespace fluxqnn
