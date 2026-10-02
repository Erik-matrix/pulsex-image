// flux-vae: the decoder of FLUX.2's VAE (AutoencoderKLFlux2) in ggml, in stages.
//
// 2026-10-02. The picture stage of flux.py was torch on the CPU: 39 s of decoding plus 17 s of imports for a 512 px
// picture - longer than the transformer on the NPU. This is the same decoder as zimage_vae_stream.cpp (FLUX.1's AE has
// the same shape: conv_in, mid = resnet + attention + resnet, four levels of three resnets with nearest upsampling,
// norm + silu + conv_out) under diffusers' tensor names, with what FLUX.2 adds:
//   * the latents come PACKED from the transformer ([Hp*Wp x 128]); they are de-normalised with the batch-norm
//     statistics stored in the model (x * sqrt(var + eps) + mean per channel) and un-patchified (128 = 32 channels x
//     2x2) to [32, 2Hp, 2Wp] here, as the pipeline does;
//   * post_quant_conv (1x1) before conv_in; the mid attention's q/k/v/out are linear layers.
// Seven stages, each with its own context and graph, the activation goes through the host in between (a single graph
// does not fit: ggml does not reuse memory along a long graph). ggml_backend_sched with the device + CPU.
//   flux-vae <vae-decoder.gguf> <lat_out.f32> <Hp> <Wp> <out_img.f32> [device=HTP0|CPU]
// out_img.f32: [3, 16*Hp, 16*Wp] float32, values around -1..1 (the caller maps them to 0..255).

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }

template <typename T> static std::vector<T> read_bin(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + p);
    _fseeki64(f, 0, SEEK_END); long long sz = _ftelli64(f); _fseeki64(f, 0, SEEK_SET);
    std::vector<T> v((size_t) (sz / sizeof(T)));
    const size_t got = fread(v.data(), sizeof(T), v.size(), f);
    fclose(f);
    if (got != v.size()) throw std::runtime_error("short read " + p);
    return v;
}

struct Vae {
    gguf_context * g = nullptr;
    FILE * gf = nullptr;
    size_t data_base = 0;
    ggml_backend_t dev = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    int groups = 32;
    float eps = 1e-6f;                      // GroupNorm eps of the decoder (diffusers default)
    bool verbose = true;
    bool has_dev = false;
    bool direct = false;                    // CPU: ggml_conv_2d_direct (batched im2col + GEMM) instead of one big im2col; set in run()

    void open(const std::string & path, const std::string & device) {
        gguf_init_params gp = { true, nullptr };
        g = gguf_init_from_file(path.c_str(), gp);
        if (!g) throw std::runtime_error("cannot read " + path);
        gf = fopen(path.c_str(), "rb");
        data_base = gguf_get_data_offset(g);
        { int64_t i = gguf_find_key(g, "flux2vae.norm_num_groups"); if (i >= 0) groups = (int) gguf_get_val_u32(g, i); }
        cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        if (!cpu) throw std::runtime_error("no CPU backend");
        ggml_backend_cpu_set_n_threads(cpu, env_int("FLUX_CPU_THREADS", 8));
        has_dev = device != "CPU";
        if (has_dev) {
            ggml_backend_dev_t d = ggml_backend_dev_by_name(device.c_str());
            if (!d) throw std::runtime_error("no device " + device);
            dev = ggml_backend_dev_init(d, nullptr);
            if (!dev) throw std::runtime_error("cannot open " + device);
        }
        ggml_backend_t bk[2] = { dev, cpu };
        sched = has_dev ? ggml_backend_sched_new(bk, nullptr, 2, 1024, false, true) : ggml_backend_sched_new(&cpu, nullptr, 1, 1024, false, true);
        if (!sched) throw std::runtime_error("no sched");
    }

    std::vector<float> host(const std::string & name) {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (id < 0 || gguf_get_tensor_type(g, id) != GGML_TYPE_F32) throw std::runtime_error("missing f32 " + name);
        std::vector<float> v(gguf_get_tensor_size(g, id) / sizeof(float));
        _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
        if (fread(v.data(), sizeof(float), v.size(), gf) != v.size()) throw std::runtime_error("short read " + name);
        return v;
    }

    // one stage: its own graph, only its own weights; the result comes back to the host with its shape
    std::vector<float> run_stage(int stage, const std::vector<float> & in, int64_t W, int64_t H, int64_t C, int64_t * oW, int64_t * oH, int64_t * oC) {
        ggml_init_params cp = { ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false), nullptr, true };
        ggml_context * ctx = ggml_init(cp);
        ggml_init_params wp = { ggml_tensor_overhead() * 128, nullptr, true };
        ggml_context * wctx = ggml_init(wp);          // the weights: their own context, their own buffer
        std::vector<std::pair<ggml_tensor *, std::string>> to_up;
        auto T = [&](const std::string & name) -> ggml_tensor * {
            const int64_t id = gguf_find_tensor(g, name.c_str());
            if (id < 0) throw std::runtime_error("missing " + name);
            ggml_tensor * t = ggml_new_tensor(wctx, gguf_get_tensor_type(g, id), GGML_MAX_DIMS, gguf_get_tensor_ne(g, id));
            ggml_set_name(t, name.c_str());
            to_up.push_back({ t, name });
            return t;
        };
        auto conv = [&](ggml_tensor * x, const std::string & pfx, int k) -> ggml_tensor * {
            ggml_tensor * w = T(pfx + ".weight");
            ggml_tensor * b = T(pfx + ".bias");
            ggml_tensor * y = direct ? ggml_conv_2d_direct(ctx, w, x, 1, 1, k / 2, k / 2, 1, 1) : ggml_conv_2d(ctx, w, x, 1, 1, k / 2, k / 2, 1, 1);
            return ggml_add(ctx, y, ggml_reshape_4d(ctx, b, 1, 1, y->ne[2], 1));
        };
        auto gn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
            const int64_t c = x->ne[2];
            ggml_tensor * y = ggml_group_norm(ctx, x, groups, eps);
            y = ggml_mul(ctx, y, ggml_reshape_4d(ctx, T(pfx + ".weight"), 1, 1, c, 1));
            return ggml_add(ctx, y, ggml_reshape_4d(ctx, T(pfx + ".bias"), 1, 1, c, 1));
        };
        auto resnet = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {
            ggml_tensor * h = conv(ggml_silu(ctx, gn(x, pfx + ".norm1")), pfx + ".conv1", 3);
            h = conv(ggml_silu(ctx, gn(h, pfx + ".norm2")), pfx + ".conv2", 3);
            ggml_tensor * s = x;
            if (gguf_find_tensor(g, (pfx + ".conv_shortcut.weight").c_str()) >= 0) s = conv(x, pfx + ".conv_shortcut", 1);
            return ggml_add(ctx, s, h);
        };
        auto attn = [&](ggml_tensor * x, const std::string & pfx) -> ggml_tensor * {   // one head, linear q/k/v/out
            const int64_t w = x->ne[0], h_ = x->ne[1], c = x->ne[2], hw = w * h_;
            ggml_tensor * h = gn(x, pfx + ".group_norm");
            ggml_tensor * t = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_2d(ctx, h, hw, c), 1, 0, 2, 3));   // [c, hw]
            auto lin = [&](const std::string & nm, ggml_tensor * a) { return ggml_add(ctx, ggml_mul_mat(ctx, T(pfx + nm + ".weight"), a), T(pfx + nm + ".bias")); };
            ggml_tensor * q = lin(".to_q", t), * k = lin(".to_k", t), * v = lin(".to_v", t);
            ggml_tensor * sc = ggml_soft_max(ctx, ggml_scale(ctx, ggml_mul_mat(ctx, k, q), 1.0f / sqrtf((float) c)));   // [hw_k, hw_q]
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));                                      // [hw, c]
            ggml_tensor * o = lin(".to_out.0", ggml_mul_mat(ctx, vt, sc));                                             // [c, hw]
            o = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, o, 1, 0, 2, 3)), w, h_, c, 1);
            return ggml_add(ctx, x, o);
        };
        auto level = [&](ggml_tensor * t, int ub, bool up) {
            const std::string p = "decoder.up_blocks." + std::to_string(ub);
            for (int b = 0; b < 3; b++) t = resnet(t, p + ".resnets." + std::to_string(b));
            if (up) t = conv(ggml_upscale(ctx, t, 2, GGML_SCALE_MODE_NEAREST), p + ".upsamplers.0.conv", 3);
            return t;
        };

        ggml_tensor * x = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W, H, C, 1);
        ggml_set_input(x);
        ggml_tensor * y = nullptr;
        switch (stage) {
            case 0: y = conv(conv(x, "post_quant_conv", 1), "decoder.conv_in", 3); break;
            case 1: y = resnet(x, "decoder.mid_block.resnets.0");
                    y = attn(y, "decoder.mid_block.attentions.0");
                    y = resnet(y, "decoder.mid_block.resnets.1"); break;
            case 2: y = level(x, 0, true);  break;
            case 3: y = level(x, 1, true);  break;
            case 4: y = level(x, 2, true);  break;
            case 5: y = level(x, 3, false); break;
            case 6: y = conv(ggml_silu(ctx, gn(x, "decoder.conv_norm_out")), "decoder.conv_out", 3); break;
            default: throw std::runtime_error("unknown stage");
        }
        ggml_set_output(y);

        ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, has_dev ? dev : cpu);
        if (!wbuf) throw std::runtime_error("stage " + std::to_string(stage) + ": cannot allocate the weights");
        ggml_backend_buffer_set_usage(wbuf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        {
            std::vector<uint8_t> buf;
            for (auto & pr : to_up) {
                const int64_t id = gguf_find_tensor(g, pr.second.c_str());
                const size_t sz = gguf_get_tensor_size(g, id);
                buf.resize(sz);
                _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id)), SEEK_SET);
                if (fread(buf.data(), 1, sz, gf) != sz) throw std::runtime_error("short read " + pr.second);
                ggml_backend_tensor_set(pr.first, buf.data(), 0, sz);
            }
        }
        ggml_cgraph * gr = ggml_new_graph_custom(ctx, 512, false);
        ggml_build_forward_expand(gr, y);
        ggml_backend_sched_reset(sched);
        if (!ggml_backend_sched_alloc_graph(sched, gr)) throw std::runtime_error("stage " + std::to_string(stage) + ": alloc failed");
        ggml_backend_tensor_set(x, in.data(), 0, in.size() * sizeof(float));
        const double t0 = now_s();
        if (ggml_backend_sched_graph_compute(sched, gr) != GGML_STATUS_SUCCESS) throw std::runtime_error("stage " + std::to_string(stage) + ": compute failed");
        const double t1 = now_s();
        std::vector<float> out((size_t) ggml_nelements(y));
        ggml_backend_tensor_get(y, out.data(), 0, out.size() * sizeof(float));
        *oW = y->ne[0]; *oH = y->ne[1]; *oC = y->ne[2];
        if (verbose)
            fprintf(stderr, "[vae] stage %d: [%lld,%lld,%lld] -> [%lld,%lld,%lld]  %d nodes, %d splits, %.2f s\n", stage, (long long) W, (long long) H,
                    (long long) C, (long long) *oW, (long long) *oH, (long long) *oC, ggml_graph_n_nodes(gr), ggml_backend_sched_get_n_splits(sched), t1 - t0);
        ggml_backend_sched_reset(sched);
        ggml_backend_buffer_free(wbuf);
        ggml_free(wctx);
        ggml_free(ctx);
        return out;
    }

    void close() {
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
        if (dev) { ggml_backend_free(dev); dev = nullptr; }
        if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
        if (gf) { fclose(gf); gf = nullptr; }
        if (g) { gguf_free(g); g = nullptr; }
    }
    ~Vae() { close(); }                     // every way out closes the NPU session
};

static int run(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: flux-vae <vae-decoder.gguf> <lat_out.f32> <Hp> <Wp> <out_img.f32> [device=HTP0|CPU]\n"); return 2; }
    const std::string model = argv[1], lat_path = argv[2], out_path = argv[5], device = argc > 6 ? argv[6] : "HTP0";
    const int Hp = atoi(argv[3]), Wp = atoi(argv[4]);
    const double T0 = now_s();
    ggml_time_init();
    ggml_backend_load_all();

    Vae V;
    V.verbose = env_int("FLUX_VAE_QUIET", 0) == 0;
    V.open(model, device);
    // Which convolution on the CPU. Measured at 512 x 512 (10-02): one big im2col + matmul 15.7 s, batched direct
    // conv 19.1 s - but the big im2col is 1.2 GB there and grows with the picture (4.8 GB at 1024 x 1024, more than
    // this PC has free). So: im2col up to 768 x 512, direct above. FLUX_VAE_CONV=im2col|direct forces either.
    if (!V.has_dev) {
        const char * e = getenv("FLUX_VAE_CONV");
        V.direct = (e && *e) ? std::string(e) == "direct" : (int64_t) Hp * Wp > 1536;
    }
    const std::vector<float> lat = read_bin<float>(lat_path);
    if (lat.size() != (size_t) Hp * Wp * 128) throw std::runtime_error("lat_out.f32 has the wrong size for " + std::to_string(Hp) + "x" + std::to_string(Wp));

    // de-normalise (batch-norm statistics over the 128 packed channels) and un-patchify: channel c*4 + i*2 + j of
    // token (h, w) is latent channel c at (2h + i, 2w + j)
    float bn_eps = 1e-4f;
    { int64_t i = gguf_find_key(V.g, "flux2vae.batch_norm_eps"); if (i >= 0) bn_eps = gguf_get_val_f32(V.g, i); }
    const std::vector<float> mean = V.host("bn.running_mean"), var = V.host("bn.running_var");
    const int64_t LH = 2 * Hp, LW = 2 * Wp;
    std::vector<float> h((size_t) 32 * LH * LW);
    for (int ty = 0; ty < Hp; ty++)
        for (int tx = 0; tx < Wp; tx++) {
            const float * tok = &lat[((size_t) ty * Wp + tx) * 128];
            for (int ch = 0; ch < 128; ch++) {
                const int c = ch / 4, i = (ch / 2) % 2, j = ch % 2;
                h[((size_t) c * LH + (2 * ty + i)) * LW + (2 * tx + j)] = tok[ch] * sqrtf(var[ch] + bn_eps) + mean[ch];
            }
        }

    int64_t W = LW, H = LH, C = 32;
    for (int s = 0; s <= 6; s++) {
        int64_t oW, oH, oC;
        h = V.run_stage(s, h, W, H, C, &oW, &oH, &oC);
        W = oW; H = oH; C = oC;
    }
    FILE * f = fopen(out_path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + out_path);
    fwrite(h.data(), sizeof(float), h.size(), f);
    fclose(f);
    double mx = 0; int bad = 0;
    for (float v : h) { mx = fmax(mx, fabs(v)); if (!std::isfinite(v)) bad++; }
    fprintf(stderr, "[vae] %s: [%lld,%lld,%lld] maxabs %.3f non-finite %d  %.2f s\n", device.c_str(), (long long) W, (long long) H, (long long) C, mx, bad, now_s() - T0);
    printf("DONE %lld %lld\n", (long long) W, (long long) H);
    fflush(stdout);
    return bad ? 1 : 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[vae] stopped: %s\n", e.what());
        return 1;
    }
}
