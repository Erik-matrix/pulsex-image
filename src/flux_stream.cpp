// flux-stream: FLUX.2 klein 9B (distilled, 4 steps) transformer on the Hexagon NPU, blocks streamed through a ring.
//
// 2026-10-02: FLUX.2 klein 9B in Q4_0 / Q8_0 on the Hexagon NPU.
// Same recipe as ltx-stream (pack file + two weight slots + backend sched) with the graph idioms of zimage-dit-stream
// (views into a fused qkv, RMS_NORM*w, ggml_rope_multi MSECT, swiglu_split, flash attention with f32 K/V views):
//   * 8 double blocks (image and text stream with their own weights, ONE attention over [text, image]) and
//     24 single blocks (one stream, attention and MLP in parallel) live in <gguf>.hexpack and are copied into one of
//     two NPU slots per block;
//   * the once-per-step parts run on the host in f32: timestep embedding, the three SHARED modulations (FLUX.2 has no
//     per-block modulation weights), img_in, final layer; txt_in once per picture.
// The math mirrors diffusers' Flux2Transformer2DModel (transformer_flux2.py) op for op; the reference that this engine
// is checked against is flux_dit/ref_flux2.py, which runs diffusers' own block modules on the bf16 weights.
//   double:  h = LN(x)*(1+scale1)+shift1 per stream; qkv = Wqkv h; q,k = rms(.)*w, rope; attention over [txt, img];
//            x += gate1 * Wproj a;  h = LN(x)*(1+scale2)+shift2;  x += gate2 * W2 (silu(Wg h) * Wv h)
//   single:  h = LN(x)*(1+scale)+shift;  a = attention(Wqkv h);  m = silu(Wg h) * Wv h;  x += gate * (Wa a + Wm m)
//   rope:    4 axes x 32 dims (t, h, w, l), theta 2000, adjacent pairs; text tokens sit at (0,0,0,l), image at (0,h,w,0)
// The GGUF is the SPLIT layout of flux_dit/split_flux2_gguf.py (mlp.0 -> 0g/0v, linear1 -> qkv/mlp_g/mlp_v,
// linear2 -> out_a/out_m): every matrix is then a plain mul_mat with a contiguous result.
// ⚠ The text stream carries values up to ~183000 after double block 7 (ref_flux2 taps): the residual is f32 end to end
// here, never f16.
//
// Workdir in:  meta.txt "Hp Wp n_text", text.f32 [n_text x 12288] (Qwen3 layers 9/18/27, ALL 512 positions - the model
//              sees the padding), noise.f32 [Hp*Wp x 128] packed latent, sigmas.txt (shifted sigmas, last one 0).
//              optional: ref.f32 [Sr x 128] + ref_meta.txt "Hr Wr" per reference picture (editing: the references'
//              latents ride along as extra image tokens at time coordinate 10, 20, ...; only the noise is stepped).
// Workdir out: lat_out.f32 [Hp*Wp x 128]; FLUX_DUMP=1: v_<i>.f32, x_<i>.f32; FLUX_TAP=1: tap_d00..s23.f32 of step 0
//              ([n_text + Hp*Wp] x 4096, text first - the same layout as the reference taps).
//   flux-stream <model-q4_0-split.gguf> <workdir> [device=HTP0|CPU]
// device CPU runs the SAME graphs on ggml's CPU backend from the raw GGUF tensors (no pack): the check that the graph
// is right before the NPU is asked.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <sys/stat.h>
#include <algorithm>
#if defined(__ARM_NEON) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename T> static std::vector<T> read_bin(const std::string & p) {
    std::vector<T> v;
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) return v;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    v.resize((size_t) n / sizeof(T));
    if (fread(v.data(), sizeof(T), v.size(), f) != v.size()) v.clear();
    fclose(f);
    return v;
}

static void write_bin(const std::string & p, const float * d, size_t n) {
    FILE * f = fopen(p.c_str(), "wb");
    if (f) { fwrite(d, sizeof(float), n, f); fclose(f); }
}

static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }

// ---- host linear algebra (f32, no bias anywhere in FLUX.2) -------------------------------------------
// y[n][o] = sum_k W[o][k] x[n][k]. 7 threads, each owning a slice of the output rows; tokens in blocks of 8 so the
// weights are streamed once per block, not once per token. Every output is one dot product: the result does not
// depend on the thread count.
static float dot(const float * a, const float * b, int n) {
    int k = 0;
    float s = 0;
#if defined(__ARM_NEON) || defined(_M_ARM64)
    float32x4_t v0 = vdupq_n_f32(0), v1 = v0, v2 = v0, v3 = v0;
    for (; k + 16 <= n; k += 16) {
        v0 = vfmaq_f32(v0, vld1q_f32(a + k), vld1q_f32(b + k));
        v1 = vfmaq_f32(v1, vld1q_f32(a + k + 4), vld1q_f32(b + k + 4));
        v2 = vfmaq_f32(v2, vld1q_f32(a + k + 8), vld1q_f32(b + k + 8));
        v3 = vfmaq_f32(v3, vld1q_f32(a + k + 12), vld1q_f32(b + k + 12));
    }
    s = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
#endif
    for (; k < n; k++) s += a[k] * b[k];
    return s;
}
static void linear(const float * x, int n, int in, const float * W, int out, float * y) {
    const int nth = (int64_t) n * out >= 4096 && out >= 7 ? 7 : 1;
    const int TB = 8;                      // tokens per pass over the weights
    auto work = [&](int o0, int o1) {
        for (int t0 = 0; t0 < n; t0 += TB) {
            const int t1 = t0 + TB < n ? t0 + TB : n;
            for (int o = o0; o < o1; o++) {
                const float * wr = W + (size_t) o * in;
                for (int t = t0; t < t1; t++) y[(size_t) t * out + o] = dot(wr, x + (size_t) t * in, in);
            }
        }
    };
    if (nth == 1) { work(0, out); return; }
    std::vector<std::thread> th;
    for (int i = 0; i < nth; i++) th.emplace_back(work, (int) ((int64_t) out * i / nth), (int) ((int64_t) out * (i + 1) / nth));
    for (auto & t : th) t.join();
}
static float silu(float v) { return v / (1.0f + expf(-v)); }

// ---- pack cache (same file format as ltx_stream.cpp / zimage_stream.cpp) ----------------------------
struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

// tensors of one double block (per stream) and of one single block, in the order the graphs index them
static const char * DBL_T[] = { "attn.qkv.weight", "attn.proj.weight", "mlp.0g.weight", "mlp.0v.weight", "mlp.2.weight",
                                "attn.norm.query_norm.scale", "attn.norm.key_norm.scale" };
static const char * SGL_T[] = { "qkv.weight", "mlp_g.weight", "mlp_v.weight", "out_a.weight", "out_m.weight",
                                "norm.query_norm.scale", "norm.key_norm.scale" };
enum { W_QKV = 0, W_PROJ = 1, W_G = 2, W_V = 3, W_MLP2 = 4, W_QN = 5, W_KN = 6, N_STREAM_T = 7 };
enum { S_QKV = 0, S_G = 1, S_V = 2, S_OUTA = 3, S_OUTM = 4, S_QN = 5, S_KN = 6, N_SGL_T = 7 };

// One block ahead: the worker reads block `want` into `buf` while the NPU computes the current block.
struct Prefetch {
    std::thread th;
    std::mutex mu;
    std::condition_variable cv_req, cv_done;
    int want = -1, inflight = -1, have = -1;
    bool quit = false;
    std::vector<uint8_t> buf;
    size_t off = 0;                        // where the block starts inside buf (page-aligned for unbuffered reads)
    FILE * f = nullptr;                    // its own handle on the pack: its own file position
    HANDLE h = INVALID_HANDLE_VALUE;       // the same file opened unbuffered (FLUX_PF_DIRECT, default)
};

// off and size multiples of 4096, dst page-aligned
static bool read_direct(HANDLE h, uint64_t off, uint8_t * dst, size_t size) {
    size_t done = 0;
    while (done < size) {
        const DWORD want = (DWORD) (size - done < ((size_t) 64 << 20) ? size - done : ((size_t) 64 << 20));
        OVERLAPPED ov{};
        ov.Offset = (DWORD) ((off + done) & 0xFFFFFFFFull);
        ov.OffsetHigh = (DWORD) ((off + done) >> 32);
        DWORD got = 0;
        if (!ReadFile(h, dst + done, want, &got, &ov) || got == 0) return false;
        done += got;
    }
    return true;
}

struct Engine {
    Prefetch pf;
    bool use_pf = false;
    size_t stage_off = 0;                  // start of the read-ahead block inside the staging buffer
    int pf_hits = 0, pf_miss = 0;
    double t_wait = 0;
    gguf_context * g = nullptr;
    FILE * gf = nullptr;
    size_t data_base = 0;
    FILE * pk = nullptr;
    ggml_backend_t npu = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool packed = true;                    // false: device CPU, raw GGUF tensors
    bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
    bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    int n_dbl = 8, n_sgl = 24, dim = 4096, heads = 32, head_dim = 128, mlp = 12288, in_ch = 128, ctx_ch = 12288, t_ch = 256;
    float theta = 2000.0f, eps = 1e-6f;
    std::vector<PackEnt> ents;             // doubles first: [il * 14 + stream * 7 + k], then singles: [n_dbl*14 + is * 7 + k]
    ggml_backend_buffer_t slot[2] = {};
    size_t slot_size = 0;
    double t_read = 0, t_copy = 0, t_comp = 0, t_host = 0, t_io = 0;
    double t_build = 0, t_alloc = 0, t_run = 0;     // inside t_comp: graph build, sched alloc, the compute call
    // FLUX_SWIGLU_F16=1: the sigmoid of SwiGLU in fp16 on the DSP (op_params[4]); a precision trade, off by default
    const int swiglu_f16 = env_int("FLUX_SWIGLU_F16", 0);
    int next_block = -1;                   // set by the step loop before each block: what to read ahead (-1 = nothing)
    std::string dir = ".";
    bool splits_logged[2] = {};

    int n_blocks() const { return n_dbl + n_sgl; }
    int n_bt(int ib) const { return ib < n_dbl ? 2 * N_STREAM_T : N_SGL_T; }
    size_t ent0(int ib) const { return ib < n_dbl ? (size_t) ib * 2 * N_STREAM_T : (size_t) n_dbl * 2 * N_STREAM_T + (size_t) (ib - n_dbl) * N_SGL_T; }
    std::string bname(int ib, int k) const {
        if (ib < n_dbl) return "double_blocks." + std::to_string(ib) + (k < N_STREAM_T ? ".img_" : ".txt_") + DBL_T[k % N_STREAM_T];
        return "single_blocks." + std::to_string(ib - n_dbl) + "." + SGL_T[k];
    }

    // a host weight, read when asked for and returned as f32 (they are F16 in the file; the big ones are used once
    // per step or once per picture, so nothing is kept)
    std::vector<float> host(const std::string & name) {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (id < 0) throw std::runtime_error("saknar " + name);
        const ggml_type ty = gguf_get_tensor_type(g, id);
        const size_t nb = gguf_get_tensor_size(g, id);
        std::vector<uint8_t> raw(nb);
        _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(raw.data(), 1, nb, gf) != nb) throw std::runtime_error("kort lasning " + name);
        std::vector<float> v;
        if (ty == GGML_TYPE_F32) { v.resize(nb / 4); memcpy(v.data(), raw.data(), nb); }
        else if (ty == GGML_TYPE_F16) { v.resize(nb / 2); ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), v.data(), (int64_t) v.size()); }
        else throw std::runtime_error(name + " ar varken F16 eller F32");
        return v;
    }

    bool open(const std::string & path, const std::string & device) {
        gguf_init_params gp = { true, nullptr };
        g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return false;
        gf = fopen(path.c_str(), "rb");
        data_base = gguf_get_data_offset(g);
        auto u32 = [&](const char * k, int & d) { int64_t i = gguf_find_key(g, k); if (i >= 0) d = (int) gguf_get_val_u32(g, i); };
        auto f32 = [&](const char * k, float & d) { int64_t i = gguf_find_key(g, k); if (i >= 0) d = gguf_get_val_f32(g, i); };
        u32("flux2.double_block_count", n_dbl); u32("flux2.single_block_count", n_sgl); u32("flux2.embedding_length", dim);
        u32("flux2.head_count", heads); u32("flux2.head_dim", head_dim); u32("flux2.feed_forward_length", mlp);
        u32("flux2.in_channels", in_ch); u32("flux2.context_length", ctx_ch); u32("flux2.time_channels", t_ch);
        f32("flux2.rope_theta", theta); f32("flux2.norm_eps", eps);
        // a SLIM file (slim_gguf.py) holds the metadata and the host tensors; its blocks exist in the .hexpack only
        const bool slim = gguf_find_key(g, "pulsex.slim") >= 0;
        if (!slim && gguf_find_tensor(g, "single_blocks.0.out_a.weight") < 0) {
            fprintf(stderr, "[flux] %s ar inte den delade layouten (kor split_flux2_gguf.py)\n", path.c_str());
            return false;
        }
        const std::string pack_path = path.substr(0, path.rfind('.')) + ".hexpack";

        packed = device != "CPU";
        cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        if (!cpu) return false;
        ggml_backend_cpu_set_n_threads(cpu, env_int("FLUX_CPU_THREADS", 8));
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(cpu);
        const char * tag = "raw";
        if (packed) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev) { fprintf(stderr, "[flux] ingen enhet %s\n", device.c_str()); return false; }
            npu = ggml_backend_dev_init(dev, nullptr);
            if (!npu) return false;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
            set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
            auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
            if (!get_packed || !set_packed || !tag_fn) { fprintf(stderr, "[flux] backenden saknar packad-API\n"); return false; }
            tag = tag_fn();
            bt = ggml_backend_get_default_buffer_type(npu);
        }
        const size_t al = ggml_backend_buft_get_alignment(bt);
        if (slim) {
            if (!packed) { fprintf(stderr, "[flux] en smal fil kan bara koras pa NPU:n (CPU-laget laser raa tensorer ur den fullstandiga GGUF:en)\n"); return false; }
            struct _stat64 st;
            if (_stat64(path.c_str(), &st) != 0) return false;
            const size_t n_want = (size_t) n_dbl * 2 * N_STREAM_T + (size_t) n_sgl * N_SGL_T;
            // the pair is recognised by the key the slim file carries (it survives a copy or a download); a slim
            // file made before 10-02 has no key and is recognised by its modification time, as it was
            const int64_t kk = gguf_find_key(g, "pulsex.pack_key");
            const uint64_t pack_key = kk >= 0 ? gguf_get_val_u64(g, kk) : (uint64_t) st.st_mtime;
            PackHdr h{};
            FILE * f = fopen(pack_path.c_str(), "rb");
            bool ok = f && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "PXRP0001", 8) && !strncmp(h.tag, tag, sizeof(h.tag)) &&
                      h.gsize == (uint64_t) st.st_size && h.gmtime == pack_key && h.n == n_want;
            if (ok) { ents.resize(n_want); ok = fread(ents.data(), sizeof(PackEnt), n_want, f) == n_want; }
            if (f) fclose(f);
            if (ok) {                                  // a pack cut short (a broken download) must not be read past its end
                struct _stat64 ps;
                uint64_t need = 0;
                for (const PackEnt & e : ents) need = e.off + e.size > need ? e.off + e.size : need;
                if (_stat64(pack_path.c_str(), &ps) != 0 || (uint64_t) ps.st_size < need) {
                    fprintf(stderr, "[flux] %s is shorter than its own table says (%llu of %llu bytes) - a broken copy or download\n", pack_path.c_str(),
                            (unsigned long long) ps.st_size, (unsigned long long) need);
                    return false;
                }
            }
            for (int ib = 0; ok && ib < n_blocks(); ib++) {
                size_t lb = 0;
                for (int k = 0; k < n_bt(ib); k++) {
                    const PackEnt & e = ents[ent0(ib) + k];
                    ok = ok && bname(ib, k) == e.name;
                    lb += GGML_PAD(e.size, al);
                }
                slot_size = lb > slot_size ? lb : slot_size;
            }
            if (!ok) {
                fprintf(stderr, "[flux] %s ar en smal fil och packfilen bredvid saknas eller passar inte (backendens tagg: %s).\n"
                                "[flux] Den kan bara byggas om fran den fullstandiga GGUF:en (kor motorn en gang pa den, sedan slim_gguf.py).\n", path.c_str(), tag);
                return false;
            }
        }
        for (int ib = 0; !slim && ib < n_blocks(); ib++) {
            size_t lb = 0;
            for (int k = 0; k < n_bt(ib); k++) {
                const std::string nm = bname(ib, k);
                const int64_t id = gguf_find_tensor(g, nm.c_str());
                if (id < 0) { fprintf(stderr, "[flux] saknar %s\n", nm.c_str()); return false; }
                ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
                ggml_context * c = ggml_init(p);
                ggml_tensor * t = ggml_new_tensor(c, gguf_get_tensor_type(g, id), GGML_MAX_DIMS, gguf_get_tensor_ne(g, id));
                PackEnt e{};
                strncpy(e.name, nm.c_str(), sizeof(e.name) - 1);
                e.type = (uint32_t) t->type;
                for (int d = 0; d < 4; d++) e.ne[d] = t->ne[d];
                e.size = ggml_backend_buft_get_alloc_size(bt, t);
                lb += GGML_PAD(e.size, al);
                ggml_free(c);
                ents.push_back(e);
            }
            slot_size = lb > slot_size ? lb : slot_size;
        }
        if (packed) {
            const uint64_t data0 = GGML_PAD(sizeof(PackHdr) + ents.size() * sizeof(PackEnt), 4096);
            if (!slim) { uint64_t off = data0; for (auto & e : ents) { e.off = off; off += GGML_PAD(e.size, 4096); } }
            struct _stat64 st;
            if (_stat64(path.c_str(), &st) != 0) return false;
            PackHdr want{};
            memcpy(want.magic, "PXRP0001", 8);
            strncpy(want.tag, tag, sizeof(want.tag) - 1);
            want.gsize = (uint64_t) st.st_size; want.gmtime = (uint64_t) st.st_mtime;
            bool ok = slim;                                              // checked above, entry table read from it
            if (FILE * f = slim ? nullptr : fopen(pack_path.c_str(), "rb")) {
                PackHdr h{};
                std::vector<PackEnt> got(ents.size());
                ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(&h, &want, offsetof(PackHdr, n)) && h.n == ents.size() &&
                     fread(got.data(), sizeof(PackEnt), got.size(), f) == got.size() &&
                     !memcmp(got.data(), ents.data(), got.size() * sizeof(PackEnt));
                fclose(f);
            }
            if (!ok && !build_pack(pack_path, want, bt)) return false;
            pk = fopen(pack_path.c_str(), "rb");
            if (!pk) return false;
            if (env_int("FLUX_PREFETCH", 1)) {
                pf.f = fopen(pack_path.c_str(), "rb");
                use_pf = pf.f != nullptr;
                if (use_pf && env_int("FLUX_PF_DIRECT", 1))
                    pf.h = CreateFileA(pack_path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                                       FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
                fprintf(stderr, "[flux] forlasning: %s\n", pf.h != INVALID_HANDLE_VALUE ? "obuffrad (direkt fran disken)" : "buffrad (fread)");
                if (use_pf) pf.th = std::thread([this] { prefetch_loop(); });
            }
        }
        for (int s = 0; s < 2; s++) {
            slot[s] = ggml_backend_buft_alloc_buffer(bt, slot_size);
            if (!slot[s]) return false;
            ggml_backend_buffer_set_usage(slot[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_t bk[2] = { npu, cpu };
        sched = packed ? ggml_backend_sched_new(bk, nullptr, 2, 4096, false, true) : ggml_backend_sched_new(&cpu, nullptr, 1, 4096, false, true);
        return sched != nullptr;
    }

    bool build_pack(const std::string & path, PackHdr h, ggml_backend_buffer_type_t bt) {
        const double t0 = now_s();
        const std::string tmp = path + ".tmp";
        FILE * f = fopen(tmp.c_str(), "wb");
        if (!f) return false;
        uint64_t maxsz = 0;
        for (auto & e : ents) maxsz = e.size > maxsz ? e.size : maxsz;
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(bt, (size_t) maxsz);
        if (!b) { fclose(f); remove(tmp.c_str()); return false; }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        h.n = (uint32_t) ents.size();
        fwrite(&h, sizeof h, 1, f);
        fwrite(ents.data(), sizeof(PackEnt), ents.size(), f);
        std::vector<uint8_t> raw, pkd;
        bool good = true;
        for (size_t i = 0; i < ents.size() && good; i++) {
            ggml_init_params p = { ggml_tensor_overhead() * 2, nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) ents[i].type, GGML_MAX_DIMS, ents[i].ne);
            ggml_tallocr ta = ggml_tallocr_new(b);
            good = ggml_tallocr_alloc(&ta, t) == GGML_STATUS_SUCCESS;
            const int64_t id = gguf_find_tensor(g, ents[i].name);
            const size_t rsz = gguf_get_tensor_size(g, id);
            raw.resize(rsz);
            _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
            good = good && fread(raw.data(), 1, rsz, gf) == rsz;
            if (good) ggml_backend_tensor_set(t, raw.data(), 0, rsz);
            pkd.assign((size_t) GGML_PAD(ents[i].size, 4096), 0);
            good = good && get_packed(t, pkd.data(), (size_t) ents[i].size);
            _fseeki64(f, (long long) ents[i].off, SEEK_SET);
            good = good && fwrite(pkd.data(), 1, pkd.size(), f) == pkd.size();
            ggml_free(c);
        }
        ggml_backend_buffer_free(b);
        good = fclose(f) == 0 && good;
        if (good) remove(path.c_str());
        if (!good || rename(tmp.c_str(), path.c_str()) != 0) { remove(tmp.c_str()); return false; }
        fprintf(stderr, "[flux] pack byggd: %zu tensorer, %.1f s -> %s\n", ents.size(), now_s() - t0, path.c_str());
        return true;
    }

    // the byte range of block ib in the pack (its tensors are consecutive)
    void block_range(int ib, uint64_t & off, size_t & size) const {
        const PackEnt & a = ents[ent0(ib)], & z = ents[ent0(ib) + n_bt(ib) - 1];
        off = a.off;
        size = (size_t) (z.off + z.size - a.off);
    }
    void prefetch_loop() {
        std::vector<uint8_t> tmp;
        std::unique_lock<std::mutex> lk(pf.mu);
        for (;;) {
            pf.cv_req.wait(lk, [&] { return pf.quit || pf.want >= 0; });
            if (pf.quit) return;
            const int ib = pf.want;
            pf.want = -1;
            pf.inflight = ib;
            lk.unlock();
            uint64_t off; size_t size;
            block_range(ib, off, size);
            const size_t padded = (size + 4095) & ~(size_t) 4095;
            if (tmp.size() < padded + 4096) tmp.resize(padded + 4096);
            const size_t al = (size_t) ((4096 - ((uintptr_t) tmp.data() & 4095)) & 4095);
            bool good;
            if (pf.h != INVALID_HANDLE_VALUE) {
                good = read_direct(pf.h, off, tmp.data() + al, padded);
            } else {
                _fseeki64(pf.f, (long long) off, SEEK_SET);
                good = fread(tmp.data() + al, 1, size, pf.f) == size;
            }
            lk.lock();
            if (good) { pf.buf.swap(tmp); pf.off = al; pf.have = ib; } else pf.have = -1;
            pf.inflight = -1;
            pf.cv_done.notify_all();
        }
    }
    void prefetch(int ib) {                 // ask for block ib; returns at once
        if (!use_pf) return;
        std::lock_guard<std::mutex> lk(pf.mu);
        if (pf.have == ib || pf.inflight == ib) return;
        pf.want = ib;
        pf.cv_req.notify_one();
    }

    // block ib -> slot s; T[k] in the order of DBL_T (img 0..6, txt 7..13) or SGL_T
    ggml_context * load_block(int ib, int s, ggml_tensor ** T, std::vector<uint8_t> & stage) {
        const int nb = n_bt(ib);
        bool pre = false;                  // stage holds the whole block, read ahead by the worker
        uint64_t off0 = 0;
        if (use_pf) {
            const double tw = now_s();
            size_t size;
            block_range(ib, off0, size);
            std::unique_lock<std::mutex> lk(pf.mu);
            pf.cv_done.wait(lk, [&] { return pf.inflight != ib; });
            if (pf.have == ib) { stage.swap(pf.buf); stage_off = pf.off; pf.have = -1; pre = true; pf_hits++; } else pf_miss++;
            lk.unlock();
            t_wait += now_s() - tw;
        }
        ggml_init_params p = { ggml_tensor_overhead() * (nb + 4), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tallocr ta = ggml_tallocr_new(slot[s]);
        for (int k = 0; k < nb; k++) {
            const PackEnt & e = ents[ent0(ib) + k];
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) e.type, GGML_MAX_DIMS, e.ne);
            ggml_set_name(t, e.name);
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) { ggml_free(c); return nullptr; }
            const double t0 = now_s();
            if (packed && pre) {
                if (!set_packed(t, stage.data() + stage_off + (size_t) (e.off - off0), (size_t) e.size)) { ggml_free(c); return nullptr; }
                t_copy += now_s() - t0;
            } else if (packed) {
                stage.resize((size_t) e.size);
                _fseeki64(pk, (long long) e.off, SEEK_SET);
                if (fread(stage.data(), 1, (size_t) e.size, pk) != e.size) { ggml_free(c); return nullptr; }
                const double t1 = now_s();
                if (!set_packed(t, stage.data(), (size_t) e.size)) { ggml_free(c); return nullptr; }
                t_read += t1 - t0; t_copy += now_s() - t1;
            } else {
                const int64_t id = gguf_find_tensor(g, e.name);
                const size_t rsz = gguf_get_tensor_size(g, id);
                stage.resize(rsz);
                _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
                if (fread(stage.data(), 1, rsz, gf) != rsz) { ggml_free(c); return nullptr; }
                const double t1 = now_s();
                ggml_backend_tensor_set(t, stage.data(), 0, rsz);
                t_read += t1 - t0; t_copy += now_s() - t1;
            }
            T[k] = t;
        }
        return c;
    }

    // ---- graph pieces shared by both block kinds --------------------------------------------------
    struct G {
        ggml_context * c;
        int dim, heads, hd;
        float eps;
        ggml_tensor * ff;
        float theta;
        ggml_tensor * ln_mod(ggml_tensor * x, ggml_tensor * scale1p, ggml_tensor * shift) {   // LN(x) * (1 + scale) + shift
            return ggml_add(c, ggml_mul(c, ggml_norm(c, x, eps), scale1p), shift);
        }
        // q or k of one stream out of its fused qkv [3*dim, S]: view -> rms * w -> rope; result [hd, heads, S] contiguous
        ggml_tensor * qk(ggml_tensor * qkv, int which, ggml_tensor * w, ggml_tensor * ids) {
            const int64_t S = qkv->ne[1];
            ggml_tensor * v = ggml_view_3d(c, qkv, hd, heads, S, (size_t) hd * sizeof(float), qkv->nb[1], (size_t) which * dim * sizeof(float));
            v = ggml_mul(c, ggml_rms_norm(c, v, eps), w);
            int sections[4] = { hd / 8, hd / 8, hd / 8, hd / 8 };          // pairs per axis: 4 axes of hd/4 dims
            return ggml_rope_multi(c, v, ids, ff, hd, sections, GGML_ROPE_TYPE_MSECT, 0, theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        }
        ggml_tensor * attention(ggml_tensor * q, ggml_tensor * k, ggml_tensor * v) {   // all [hd, heads, S]
            const int64_t S = q->ne[2];
            static const int kv_f16 = env_int("FLUX_KV_F16", 0);
            static const int fast_exp2 = env_int("FLUX_FA_FAST_EXP2", 0);   // the exact softmax until measured here
            q = ggml_permute(c, q, 0, 2, 1, 3);
            k = ggml_permute(c, k, 0, 2, 1, 3);
            v = ggml_permute(c, v, 0, 2, 1, 3);
            if (kv_f16) { k = ggml_cast(c, k, GGML_TYPE_F16); v = ggml_cast(c, v, GGML_TYPE_F16); }
            ggml_tensor * a = ggml_flash_attn_ext(c, q, k, v, nullptr, 1.0f / sqrtf((float) hd), 0.0f, 0.0f);
            if (fast_exp2) a->op_params[4] |= 1;
            ggml_prec_set_acc(a, GGML_PREC_F32);
            return ggml_reshape_2d(c, a, dim, S);
        }
    };

    bool compute(ggml_cgraph * gr, int kind) {
        if (ggml_backend_sched_graph_compute(sched, gr) != GGML_STATUS_SUCCESS) return false;
        if (!splits_logged[kind]) {
            splits_logged[kind] = true;
            fprintf(stderr, "[flux] %s-graf: %d noder, %d delningar\n", kind == 0 ? "dubbel" : "enkel", ggml_graph_n_nodes(gr), ggml_backend_sched_get_n_splits(sched));
        }
        return true;
    }

    // one double block: img [Si x dim], txt [St x dim] on the host, in and out. mod_*: 6 x dim (shift1, scale1,
    // gate1, shift2, scale2, gate2) with "1 +" already added to the two scale vectors. ids_*: [4 x S] axis-major.
    bool run_double(int ib, int s, std::vector<float> & img, int Si, std::vector<float> & txt, int St, const float * mod_img,
                    const float * mod_txt, const std::vector<int32_t> & ids_img, const std::vector<int32_t> & ids_txt,
                    const std::vector<float> & ffv, std::vector<uint8_t> & stage, ggml_context *& wctx) {
        if (wctx) { ggml_free(wctx); wctx = nullptr; }
        ggml_tensor * W[2 * N_STREAM_T] = {};
        wctx = load_block(ib, s, W, stage);
        if (!wctx) { fprintf(stderr, "[flux] block %d laddades inte\n", ib); return false; }
        if (next_block >= 0) prefetch(next_block);
        const double tc = now_s();
        ggml_init_params p = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * x[2]   = { ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, Si), ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, St) };
        ggml_tensor * ids[2] = { ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * Si), ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * St) };
        ggml_tensor * ff = ggml_new_tensor_1d(c, GGML_TYPE_F32, head_dim / 2);
        ggml_tensor * m[2][6];
        for (int a = 0; a < 2; a++) {
            ggml_set_input(x[a]); ggml_set_input(ids[a]);
            for (int k = 0; k < 6; k++) { m[a][k] = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim); ggml_set_input(m[a][k]); }
        }
        ggml_set_input(ff);
        G gph{ c, dim, heads, head_dim, eps, ff, theta };

        ggml_tensor * qkv[2], * q[2], * k[2], * v[2];
        for (int a = 0; a < 2; a++) {                                    // 0 image, 1 text
            ggml_tensor ** Wa = W + a * N_STREAM_T;
            ggml_tensor * h = gph.ln_mod(x[a], m[a][1], m[a][0]);
            qkv[a] = ggml_mul_mat(c, Wa[W_QKV], h);
            q[a] = gph.qk(qkv[a], 0, Wa[W_QN], ids[a]);
            k[a] = gph.qk(qkv[a], 1, Wa[W_KN], ids[a]);
            // v: each token's dim values are contiguous inside qkv -> [dim, S] view with the token stride of qkv
            v[a] = ggml_view_2d(c, qkv[a], dim, qkv[a]->ne[1], qkv[a]->nb[1], (size_t) 2 * dim * sizeof(float));
            // 10-02: CONCAT on these strided views cost 79 ms per block (17 % of a step on the DSP); contiguous first
            static const int v_cont = env_int("FLUX_V_CONT", 1);
            if (v_cont) v[a] = ggml_cont(c, v[a]);
        }
        const int S = St + Si;
        ggml_tensor * qj = ggml_concat(c, q[1], q[0], 2);                // text first, as the reference
        ggml_tensor * kj = ggml_concat(c, k[1], k[0], 2);
        ggml_tensor * vj = ggml_reshape_3d(c, ggml_concat(c, v[1], v[0], 1), head_dim, heads, S);
        ggml_tensor * att = gph.attention(qj, kj, vj);                   // [dim, S]
        ggml_tensor * a_part[2] = { ggml_view_2d(c, att, dim, Si, att->nb[1], (size_t) St * att->nb[1]),
                                    ggml_view_2d(c, att, dim, St, att->nb[1], 0) };
        ggml_tensor * out[2];
        for (int a = 0; a < 2; a++) {
            ggml_tensor ** Wa = W + a * N_STREAM_T;
            ggml_tensor * x1 = ggml_add(c, x[a], ggml_mul(c, ggml_mul_mat(c, Wa[W_PROJ], a_part[a]), m[a][2]));
            ggml_tensor * h2 = gph.ln_mod(x1, m[a][4], m[a][3]);
            ggml_tensor * act = ggml_swiglu_split(c, ggml_mul_mat(c, Wa[W_G], h2), ggml_mul_mat(c, Wa[W_V], h2));
            if (swiglu_f16) act->op_params[4] = 1;
            out[a] = ggml_add(c, x1, ggml_mul(c, ggml_mul_mat(c, Wa[W_MLP2], act), m[a][5]));
            ggml_set_output(out[a]);
        }
        ggml_cgraph * gr = ggml_new_graph_custom(c, 512, false);
        ggml_build_forward_expand(gr, out[0]);
        ggml_build_forward_expand(gr, out[1]);
        const double t_al = now_s();
        t_build += t_al - tc;
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[flux] alloc misslyckades (dubbel %d)\n", ib); return false; }
        const double t_up = now_s();
        t_alloc += t_up - t_al;
        ggml_backend_tensor_set(x[0], img.data(), 0, img.size() * sizeof(float));
        ggml_backend_tensor_set(x[1], txt.data(), 0, txt.size() * sizeof(float));
        ggml_backend_tensor_set(ids[0], ids_img.data(), 0, ids_img.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ids[1], ids_txt.data(), 0, ids_txt.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ff, ffv.data(), 0, ffv.size() * sizeof(float));
        for (int k2 = 0; k2 < 6; k2++) {
            ggml_backend_tensor_set(m[0][k2], mod_img + (size_t) k2 * dim, 0, (size_t) dim * sizeof(float));
            ggml_backend_tensor_set(m[1][k2], mod_txt + (size_t) k2 * dim, 0, (size_t) dim * sizeof(float));
        }
        const double t_c0 = now_s();
        t_io += t_c0 - t_up;
        if (!compute(gr, 0)) { ggml_free(c); fprintf(stderr, "[flux] compute misslyckades i dubbelblock %d\n", ib); return false; }
        const double t_dn = now_s();
        t_run += t_dn - t_c0;
        ggml_backend_tensor_get(out[0], img.data(), 0, img.size() * sizeof(float));
        ggml_backend_tensor_get(out[1], txt.data(), 0, txt.size() * sizeof(float));
        t_io += now_s() - t_dn;
        ggml_free(c);
        t_comp += now_s() - tc;
        return true;
    }

    // one single block: x [S x dim] (text first). mod: 3 x dim (shift, scale with "1 +", gate). ids: [4 x S] axis-major.
    bool run_single(int ib, int s, std::vector<float> & xs, int S, const float * mod, const std::vector<int32_t> & ids_all,
                    const std::vector<float> & ffv, std::vector<uint8_t> & stage, ggml_context *& wctx) {
        if (wctx) { ggml_free(wctx); wctx = nullptr; }
        ggml_tensor * W[N_SGL_T] = {};
        wctx = load_block(ib, s, W, stage);
        if (!wctx) { fprintf(stderr, "[flux] block %d laddades inte\n", ib); return false; }
        if (next_block >= 0) prefetch(next_block);
        const double tc = now_s();
        ggml_init_params p = { ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(512, false), nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tensor * x = ggml_new_tensor_2d(c, GGML_TYPE_F32, dim, S);
        ggml_tensor * ids = ggml_new_tensor_1d(c, GGML_TYPE_I32, 4 * S);
        ggml_tensor * ff = ggml_new_tensor_1d(c, GGML_TYPE_F32, head_dim / 2);
        ggml_tensor * m[3];
        for (int k = 0; k < 3; k++) { m[k] = ggml_new_tensor_1d(c, GGML_TYPE_F32, dim); ggml_set_input(m[k]); }
        for (ggml_tensor * t : { x, ids, ff }) ggml_set_input(t);
        G gph{ c, dim, heads, head_dim, eps, ff, theta };

        ggml_tensor * h = gph.ln_mod(x, m[1], m[0]);
        ggml_tensor * qkv = ggml_mul_mat(c, W[S_QKV], h);
        ggml_tensor * q = gph.qk(qkv, 0, W[S_QN], ids);
        ggml_tensor * k = gph.qk(qkv, 1, W[S_KN], ids);
        ggml_tensor * v = ggml_view_3d(c, qkv, head_dim, heads, S, (size_t) head_dim * sizeof(float), qkv->nb[1], (size_t) 2 * dim * sizeof(float));
        ggml_tensor * att = gph.attention(q, k, v);
        ggml_tensor * act = ggml_swiglu_split(c, ggml_mul_mat(c, W[S_G], h), ggml_mul_mat(c, W[S_V], h));
        if (swiglu_f16) act->op_params[4] = 1;
        ggml_tensor * o = ggml_add(c, ggml_mul_mat(c, W[S_OUTA], att), ggml_mul_mat(c, W[S_OUTM], act));
        ggml_tensor * xo = ggml_add(c, x, ggml_mul(c, o, m[2]));
        ggml_set_output(xo);
        ggml_cgraph * gr = ggml_new_graph_custom(c, 512, false);
        ggml_build_forward_expand(gr, xo);
        const double t_al = now_s();
        t_build += t_al - tc;
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[flux] alloc misslyckades (enkel %d)\n", ib - n_dbl); return false; }
        const double t_up = now_s();
        t_alloc += t_up - t_al;
        ggml_backend_tensor_set(x, xs.data(), 0, xs.size() * sizeof(float));
        ggml_backend_tensor_set(ids, ids_all.data(), 0, ids_all.size() * sizeof(int32_t));
        ggml_backend_tensor_set(ff, ffv.data(), 0, ffv.size() * sizeof(float));
        for (int k2 = 0; k2 < 3; k2++) ggml_backend_tensor_set(m[k2], mod + (size_t) k2 * dim, 0, (size_t) dim * sizeof(float));
        const double t_c0 = now_s();
        t_io += t_c0 - t_up;
        if (!compute(gr, 1)) { ggml_free(c); fprintf(stderr, "[flux] compute misslyckades i enkelblock %d\n", ib - n_dbl); return false; }
        const double t_dn = now_s();
        t_run += t_dn - t_c0;
        ggml_backend_tensor_get(xo, xs.data(), 0, xs.size() * sizeof(float));
        t_io += now_s() - t_dn;
        ggml_free(c);
        t_comp += now_s() - tc;
        return true;
    }

    // Every way out of the program goes through here (the destructor): the NPU session is closed whichever
    // return or exception ends the run. Weight contexts first, then the sched, the slots, the backends.
    ggml_context * wctx[2] = {};
    void close() {
        if (pf.th.joinable()) {
            { std::lock_guard<std::mutex> lk(pf.mu); pf.quit = true; }
            pf.cv_req.notify_one();
            pf.th.join();
        }
        if (pf.f) { fclose(pf.f); pf.f = nullptr; }
        if (pf.h != INVALID_HANDLE_VALUE) { CloseHandle(pf.h); pf.h = INVALID_HANDLE_VALUE; }
        for (auto & w : wctx) if (w) { ggml_free(w); w = nullptr; }
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
        for (auto & b : slot) if (b) { ggml_backend_buffer_free(b); b = nullptr; }
        if (npu) { ggml_backend_free(npu); npu = nullptr; fprintf(stderr, "[flux] NPU-sessionen stangd\n"); }
        if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
        if (pk) { fclose(pk); pk = nullptr; }
        if (gf) { fclose(gf); gf = nullptr; }
        if (g) { gguf_free(g); g = nullptr; }
    }
    ~Engine() { close(); }
};

static double rms_of(const std::vector<float> & v, size_t a, size_t b) {
    double s = 0;
    for (size_t i = a; i < b; i++) s += (double) v[i] * v[i];
    return sqrt(s / (double) (b - a));
}

static int run(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: flux-stream <model-q4_0-split.gguf> <workdir> [device=HTP0|CPU]\n"); return 2; }
    const std::string model = argv[1], dir = argv[2], device = argc > 3 ? argv[3] : "HTP0";
    const bool dump = env_int("FLUX_DUMP", 0) != 0, tap = env_int("FLUX_TAP", 0) != 0;
    const int max_blocks = env_int("FLUX_BLOCKS", 0);     // > 0: stop after this many blocks of step 0 (bring-up)
    const int max_steps = env_int("FLUX_STEPS", 0);       // > 0: stop after this many steps
    const double T0 = now_s();
    ggml_time_init();
    ggml_backend_load_all();

    int Hp = 0, Wp = 0, St = 0;
    { FILE * f = fopen((dir + "\\meta.txt").c_str(), "r");
      if (!f || fscanf(f, "%d %d %d", &Hp, &Wp, &St) != 3) { fprintf(stderr, "meta.txt saknas/felaktig i %s\n", dir.c_str()); return 1; }
      fclose(f); }
    std::vector<float> sig;
    { FILE * f = fopen((dir + "\\sigmas.txt").c_str(), "r");
      if (f) { float v; while (fscanf(f, "%f", &v) == 1) sig.push_back(v); fclose(f); } }
    if (sig.size() < 2) { fprintf(stderr, "sigmas.txt saknas i %s\n", dir.c_str()); return 1; }

    Engine E;
    E.dir = dir;
    if (!E.open(model, device)) { fprintf(stderr, "[flux] kunde inte oppna motorn\n"); return 1; }
    // reference pictures: ref_meta.txt "Hr Wr" per picture, ref.f32 their packed latents one after the other
    std::vector<std::pair<int, int>> ref_hw;
    int Sr = 0;
    { FILE * f = fopen((dir + "\\ref_meta.txt").c_str(), "r");
      if (f) { int rh, rw; while (fscanf(f, "%d %d", &rh, &rw) == 2) { ref_hw.push_back({ rh, rw }); Sr += rh * rw; } fclose(f); } }
    // Si = the tokens being denoised, Sa = every image token the transformer sees (noise + references)
    const int Si = Hp * Wp, Sa = Si + Sr, S = St + Sa, D = E.dim;
    fprintf(stderr, "[flux] %s: %d dubbel + %d enkel, dim %d, bild %dx%d = %d token + text %d, %zu steg, plats %.1f MB x2 | start %.2f s\n",
            device.c_str(), E.n_dbl, E.n_sgl, D, Hp, Wp, Si, St, sig.size() - 1, E.slot_size / 1048576.0, now_s() - T0);

    std::vector<float> text = read_bin<float>(dir + "\\text.f32");
    std::vector<float> lat = read_bin<float>(dir + "\\noise.f32");
    std::vector<float> ref;
    if (Sr) {
        ref = read_bin<float>(dir + "\\ref.f32");
        if (ref.size() != (size_t) Sr * E.in_ch) { fprintf(stderr, "[flux] ref.f32: %zu varden, vantade %d\n", ref.size(), Sr * E.in_ch); return 1; }
        fprintf(stderr, "[flux] %zu referensbild(er): %d token till (tidsaxel 10, 20, ...)\n", ref_hw.size(), Sr);
    }
    if (text.size() != (size_t) St * E.ctx_ch || lat.size() != (size_t) Si * E.in_ch) {
        fprintf(stderr, "[flux] fel storlek: text %zu (vantade %d), noise %zu (vantade %d)\n", text.size(), St * E.ctx_ch, lat.size(), Si * E.in_ch);
        return 1;
    }

    double th = now_s();
    // once per picture: the text through txt_in; position ids; the frequency factors of the 4-axis rope
    std::vector<float> enc0((size_t) St * D);
    { const std::vector<float> w = E.host("txt_in.weight"); linear(text.data(), St, E.ctx_ch, w.data(), D, enc0.data()); }
    std::vector<int32_t> ids_img((size_t) 4 * Sa, 0), ids_txt((size_t) 4 * St, 0), ids_all((size_t) 4 * S, 0);
    for (int h = 0; h < Hp; h++)
        for (int w = 0; w < Wp; w++) { ids_img[(size_t) Sa + h * Wp + w] = h; ids_img[(size_t) 2 * Sa + h * Wp + w] = w; }
    { int at = Si;                                                       // references: (10 + 10*i, h, w, 0)
      for (size_t r = 0; r < ref_hw.size(); r++)
          for (int h = 0; h < ref_hw[r].first; h++)
              for (int w = 0; w < ref_hw[r].second; w++, at++) {
                  ids_img[at] = 10 + 10 * (int) r; ids_img[(size_t) Sa + at] = h; ids_img[(size_t) 2 * Sa + at] = w;
              } }
    for (int l = 0; l < St; l++) ids_txt[(size_t) 3 * St + l] = l;
    for (int a = 0; a < 4; a++) {                                        // [text | image] per axis
        std::copy(ids_txt.begin() + (size_t) a * St, ids_txt.begin() + (size_t) (a + 1) * St, ids_all.begin() + (size_t) a * S);
        std::copy(ids_img.begin() + (size_t) a * Sa, ids_img.begin() + (size_t) (a + 1) * Sa, ids_all.begin() + (size_t) a * S + St);
    }
    // ggml's rope steps theta by base^(-2/head_dim) per pair; each axis wants base^(-2/axis_dim) and a restart per
    // section. ff divides theta per pair and makes up the difference exactly (as zimage_stream.cpp).
    const int HD = E.head_dim, AX = HD / 4;
    std::vector<float> ffv((size_t) HD / 2, 1.0f);
    for (int a = 0; a < 4; a++)
        for (int j = 0; j < AX / 2; j++) ffv[(size_t) a * (AX / 2) + j] = (float) pow((double) E.theta, 2.0 * (1.0 / AX - 1.0 / HD) * j);
    E.t_host += now_s() - th;

    std::vector<uint8_t> stage;
    ggml_context ** wctx = E.wctx;
    std::vector<float> img((size_t) Sa * D), txt, xs((size_t) S * D), v((size_t) Si * E.in_ch);
    // timestep embedding for EVERY step, then each of the four big host matrices is read once and applied to all of
    // them: sinusoid 256 ([cos, sin]) -> in_layer -> silu -> out_layer = temb; modulations and final take silu(temb)
    th = now_s();
    const size_t NS = sig.size() - 1;
    std::vector<float> st_all(NS * D), mi_all(NS * 6 * D), mt_all(NS * 6 * D), ms_all(NS * 3 * D), fin_all(NS * 2 * D);
    {
        const std::vector<float> w_in = E.host("time_in.in_layer.weight"), w_out = E.host("time_in.out_layer.weight");
        const int half = E.t_ch / 2;
        std::vector<float> sinus(E.t_ch), e1(D), te(D);
        for (size_t i = 0; i < NS; i++) {
            const float t = sig[i] * 1000.0f;
            for (int k = 0; k < half; k++) {
                const float fr = expf(-logf(10000.0f) * k / (float) half);
                sinus[k] = cosf(t * fr);
                sinus[half + k] = sinf(t * fr);
            }
            linear(sinus.data(), 1, E.t_ch, w_in.data(), D, e1.data());
            for (auto & q : e1) q = silu(q);
            linear(e1.data(), 1, D, w_out.data(), D, te.data());
            for (int k = 0; k < D; k++) st_all[i * D + k] = silu(te[k]);
        }
    }
    { const std::vector<float> w = E.host("double_stream_modulation_img.lin.weight"); linear(st_all.data(), (int) NS, D, w.data(), 6 * D, mi_all.data()); }
    { const std::vector<float> w = E.host("double_stream_modulation_txt.lin.weight"); linear(st_all.data(), (int) NS, D, w.data(), 6 * D, mt_all.data()); }
    { const std::vector<float> w = E.host("single_stream_modulation.lin.weight"); linear(st_all.data(), (int) NS, D, w.data(), 3 * D, ms_all.data()); }
    { const std::vector<float> w = E.host("final_layer.adaLN_modulation.1.weight"); linear(st_all.data(), (int) NS, D, w.data(), 2 * D, fin_all.data()); }
    for (size_t i = 0; i < NS; i++)
        for (int k = 0; k < D; k++) {                                    // the "1 +" of every scale, once, on the host
            mi_all[i * 6 * D + (size_t) 1 * D + k] += 1.0f; mi_all[i * 6 * D + (size_t) 4 * D + k] += 1.0f;
            mt_all[i * 6 * D + (size_t) 1 * D + k] += 1.0f; mt_all[i * 6 * D + (size_t) 4 * D + k] += 1.0f;
            ms_all[i * 3 * D + (size_t) 1 * D + k] += 1.0f;
        }
    const std::vector<float> w_img_in = E.host("img_in.weight"), w_final = E.host("final_layer.linear.weight");
    std::vector<float> ref_emb((size_t) Sr * D);                         // the references through img_in, once
    if (Sr) linear(ref.data(), Sr, E.in_ch, w_img_in.data(), D, ref_emb.data());
    E.t_host += now_s() - th;
    bool ok = true;
    for (size_t i = 0; i + 1 < sig.size() && ok; i++) {
        if (max_steps > 0 && (int) i >= max_steps) break;
        const double ts = now_s();
        th = now_s();
        const float * mod_img = &mi_all[i * 6 * D], * mod_txt = &mt_all[i * 6 * D], * mod_sgl = &ms_all[i * 3 * D], * fin = &fin_all[i * 2 * D];
        linear(lat.data(), Si, E.in_ch, w_img_in.data(), D, img.data());
        if (Sr) std::copy(ref_emb.begin(), ref_emb.end(), img.begin() + (size_t) Si * D);
        txt = enc0;
        E.t_host += now_s() - th;

        int nrun = 0;
        for (int ib = 0; ib < E.n_dbl && ok; ib++, nrun++) {
            if (max_blocks > 0 && nrun >= max_blocks) break;
            E.next_block = (max_blocks > 0 && nrun + 1 >= max_blocks) ? -1 : ib + 1;
            ok = E.run_double(ib, ib & 1, img, Sa, txt, St, mod_img, mod_txt, ids_img, ids_txt, ffv, stage, wctx[ib & 1]);
            if (ok && (tap || i == 0))
                fprintf(stderr, "[flux]   dubbel %d  bild rms %.4f  text rms %.4f\n", ib, rms_of(img, 0, img.size()), rms_of(txt, 0, txt.size()));
            if (ok && tap && i == 0) {
                std::copy(txt.begin(), txt.end(), xs.begin());
                std::copy(img.begin(), img.end(), xs.begin() + (size_t) St * D);
                char nm[32]; snprintf(nm, sizeof nm, "\\tap_d%02d.f32", ib);
                write_bin(dir + nm, xs.data(), xs.size());
            }
        }
        std::copy(txt.begin(), txt.end(), xs.begin());
        std::copy(img.begin(), img.end(), xs.begin() + (size_t) St * D);
        for (int is = 0; is < E.n_sgl && ok; is++, nrun++) {
            if (max_blocks > 0 && nrun >= max_blocks) break;
            const int ib = E.n_dbl + is;
            const bool last_step = i + 2 == sig.size() || (max_steps > 0 && (int) i + 1 >= max_steps);
            E.next_block = (max_blocks > 0 && nrun + 1 >= max_blocks) ? -1 : (ib + 1 < E.n_blocks() ? ib + 1 : (last_step ? -1 : 0));
            ok = E.run_single(ib, ib & 1, xs, S, mod_sgl, ids_all, ffv, stage, wctx[ib & 1]);
            if (ok && (tap || i == 0)) fprintf(stderr, "[flux]   enkel %2d  rms %.4f\n", is, rms_of(xs, 0, xs.size()));
            if (ok && tap && i == 0) {
                char nm[32]; snprintf(nm, sizeof nm, "\\tap_s%02d.f32", is);
                write_bin(dir + nm, xs.data(), xs.size());
            }
        }
        if (!ok || (max_blocks > 0 && nrun >= max_blocks && nrun < E.n_blocks())) break;

        th = now_s();
        // final: LN(x) * (1 + scale) + shift over the image tokens (the original gives (shift, scale)), then proj
        std::vector<float> y((size_t) Si * D);
        for (int tkn = 0; tkn < Si; tkn++) {
            const float * xr = &xs[(size_t) (St + tkn) * D];
            double m = 0, v2 = 0;
            for (int k = 0; k < D; k++) m += xr[k];
            m /= D;
            for (int k = 0; k < D; k++) v2 += (xr[k] - m) * (xr[k] - m);
            const float inv = (float) (1.0 / sqrt(v2 / D + (double) E.eps));
            float * yr = &y[(size_t) tkn * D];
            for (int k = 0; k < D; k++) yr[k] = (float) (xr[k] - m) * inv * (1.0f + fin[(size_t) D + k]) + fin[k];
        }
        linear(y.data(), Si, D, w_final.data(), E.in_ch, v.data());
        const float ds = sig[i + 1] - sig[i];
        for (size_t k = 0; k < lat.size(); k++) lat[k] += ds * v[k];
        E.t_host += now_s() - th;
        if (dump) {
            write_bin(dir + "\\v_" + std::to_string(i) + ".f32", v.data(), v.size());
            write_bin(dir + "\\x_" + std::to_string(i) + ".f32", lat.data(), lat.size());
        }
        fprintf(stderr, "[flux] steg %zu sigma %.4f -> %.4f  |v| rms %.4f  |x| rms %.4f  %.2f s\n", i, sig[i], sig[i + 1],
                rms_of(v, 0, v.size()), rms_of(lat, 0, lat.size()), now_s() - ts);
    }
    if (ok) {
        write_bin(dir + "\\lat_out.f32", lat.data(), lat.size());
        printf("[flux] klar -> lat_out.f32\n");
        fflush(stdout);
    }
    fprintf(stderr, "[flux] totalt %.1f s | block: las %.2f + vanta pa forlasning %.2f (%d traffar, %d missar) + kopiera %.2f + berakna %.2f (graf %.2f, alloc %.2f, upp/ned %.2f, compute %.2f) s | varden %.2f s\n",
            now_s() - T0, E.t_read, E.t_wait, E.pf_hits, E.pf_miss, E.t_copy, E.t_comp, E.t_build, E.t_alloc, E.t_io, E.t_run, E.t_host);
    return ok ? 0 : 1;
}

int main(int argc, char ** argv) {
    // 2026-10-09: ggml-hexagon's flash softmax drops every weight under 2^-15; P * 2^3 (floor 2^-18) put FLUX 1024^2 2.7x
    // closer to the CPU reference (latents rel L2 0.105 -> 0.039, picture 28.9 -> 37.8 dB) at no cost; 2^5 overflows fp16
    // with FLUX's values (measured 2026-10-09). Default here so flux.py, flux-make and the window all get it;
    // set GGML_HEXAGON_FA_SCALE_EXP2 yourself to override (-1 = ggml's old exp). _putenv_s, not SetEnvironmentVariable:
    // ggml reads it with getenv, which sees only the CRT's copy.
    if (!getenv("GGML_HEXAGON_FA_SCALE_EXP2")) _putenv_s("GGML_HEXAGON_FA_SCALE_EXP2", "3");
    // 2026-10-10: the same exp2 in 14 HVX ops instead of 18 (wan22\patch_ggml_fa_trunc_exp2.py + _trunc_qf.py): 1024^2
    // transformer 50.0 -> 47.8 s; vs the CPU reference 3.90e-2 -> 4.04e-2, picture 37.78 -> 37.45 dB - the size an
    // independent rounding perturbation of 9.9e-3 gives after diffusion (sqrt(3.90^2 + 0.99^2) = 4.02); on Wan's attention,
    // measured directly, 14 ops is the more exact (1.563e-3 vs 1.623e-3). tools\fa_trunc_flux_1010.py. 0 = ggml's recipe.
    if (!getenv("GGML_HEXAGON_FA_TRUNC_EXP2")) _putenv_s("GGML_HEXAGON_FA_TRUNC_EXP2", "2");
    fprintf(stderr, "[flux] GGML_HEXAGON_FA_SCALE_EXP2=%s TRUNC_EXP2=%s\n", getenv("GGML_HEXAGON_FA_SCALE_EXP2"),
            getenv("GGML_HEXAGON_FA_TRUNC_EXP2"));
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[flux] avbrott: %s\n", e.what());
        return 1;
    }
}
