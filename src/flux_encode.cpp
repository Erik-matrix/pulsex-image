// flux-encode: the text encoder of FLUX.2 klein 9B (Qwen3 8B) on the Hexagon NPU, layers streamed through two slots.
//
// 2026-10-02. The engine is the one of zimage_encode_stream.h (Qwen3-4B for Z-Image) - llama.cpp only tokenizes
// (vocab_only), the layers run one at a time in our own ggml graphs from <model>.hexpack - with what FLUX.2 asks for:
//   * THREE taps: the hidden states after layers 9, 18 and 27 (hidden_states[9|18|27] of transformers), written side
//     by side: out [n_tokens x 3*n_embd]. Only layers 0..26 run; layers 27..35 and the output head are never read.
//   * the prompt is PADDED to n_tokens (512) with <|endoftext|> and the padding is part of the result: FLUX's
//     transformer has no text mask, it was trained on all 512 positions. The attention mask is the one of
//     transformers' create_causal_mask with the padding mask: key j is open for query i iff j <= i AND j < n_real
//     (a padding position sees the prompt, never the padding, not even itself).
// The graph per layer mirrors llama.cpp's qwen3 graph (RMS norm * w, separate q/k/v with 8 KV heads, q/k RMS norm,
// NEOX rope base 1e6, flash attention with the f16 mask and f32 accumulation, swiglu_split, residuals).
// The reference this is checked against: flux_dit/ref_flux2.py text (--text-gguf for the same Q4_0 weights).
//
//   flux-encode <model.gguf> <text file> <out.f32> [device=HTP0|CPU] [n_tokens=512]
// The text file is used verbatim (chat template included - the caller writes it). stdout: "DONE <n_real>".
// FLUX_ENC_TOKENS=<file>: also write the token ids (int32 x n_tokens). FLUX_ENC_LOG=1: timings and per-layer rms.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "llama.h"

#include <sys/stat.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct PackHdr { char magic[8]; char tag[32]; uint64_t gsize, gmtime; uint32_t n, pad; };
struct PackEnt { char name[96]; uint32_t type, pad; int64_t ne[4]; uint64_t off, size; };
static_assert(sizeof(PackHdr) == 64 && sizeof(PackEnt) == 152, "packcache layout");

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int env_int(const char * k, int d) { const char * e = getenv(k); return e && *e ? atoi(e) : d; }

struct Layer { ggml_tensor *an, *wq, *wk, *wv, *qn, *kn, *wo, *fn, *wg, *wu, *wd; };
static const char * LAYER_TENSORS[11] = { "attn_norm", "attn_q", "attn_k", "attn_v", "attn_q_norm", "attn_k_norm",
                                          "attn_output", "ffn_norm", "ffn_gate", "ffn_up", "ffn_down" };

struct Engine {
    gguf_context * g = nullptr;
    FILE * gf = nullptr;
    size_t data_base = 0;
    FILE * pk = nullptr;
    ggml_backend_t htp = nullptr, cpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    bool packed = true;
    bool (*get_packed)(const ggml_tensor *, void *, size_t) = nullptr;
    bool (*set_packed)(ggml_tensor *, const void *, size_t) = nullptr;
    std::vector<PackEnt> ents;          // [layer*11 + k]
    ggml_backend_buffer_t slot[2] = {};
    ggml_context * lctx[2] = {};
    size_t slot_size = 0;
    int n_layer_run = 27;
    int64_t n_embd = 4096, n_head = 32, n_head_kv = 8, head_dim = 128;
    float eps = 1e-6f, rope_base = 1e6f;
    int n_ctx_orig = 40960;
    double t_copy = 0, t_read = 0, t_comp = 0;

    std::string tname(int il, int k) const { return "blk." + std::to_string(il) + "." + LAYER_TENSORS[k] + ".weight"; }

    bool open(const std::string & path, const std::string & device, int layers) {
        n_layer_run = layers;
        gguf_init_params gp = { true, nullptr };
        g = gguf_init_from_file(path.c_str(), gp);
        if (!g) return false;
        gf = fopen(path.c_str(), "rb");
        data_base = gguf_get_data_offset(g);
        auto u32 = [&](const char * k, int64_t & dst) { int64_t id = gguf_find_key(g, k); if (id >= 0) dst = gguf_get_val_u32(g, id); };
        auto f32 = [&](const char * k, float & dst) { int64_t id = gguf_find_key(g, k); if (id >= 0) dst = gguf_get_val_f32(g, id); };
        u32("qwen3.embedding_length", n_embd);
        u32("qwen3.attention.head_count", n_head);
        u32("qwen3.attention.head_count_kv", n_head_kv);
        u32("qwen3.attention.key_length", head_dim);
        f32("qwen3.attention.layer_norm_rms_epsilon", eps);
        f32("qwen3.rope.freq_base", rope_base);
        { int64_t c = n_ctx_orig; u32("qwen3.context_length", c); n_ctx_orig = (int) c; }

        packed = device != "CPU";
        cpu = ggml_backend_dev_init(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr);
        if (!cpu) return false;
        ggml_backend_cpu_set_n_threads(cpu, env_int("FLUX_CPU_THREADS", 8));
        ggml_backend_buffer_type_t bt = ggml_backend_get_default_buffer_type(cpu);
        const char * tag = "raw";
        if (packed) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (!dev) { fprintf(stderr, "[enc] no device %s\n", device.c_str()); return false; }
            htp = ggml_backend_dev_init(dev, nullptr);
            if (!htp) return false;
            ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
            get_packed = (bool (*)(const ggml_tensor *, void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_get_tensor_packed");
            set_packed = (bool (*)(ggml_tensor *, const void *, size_t)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_set_tensor_packed");
            auto tag_fn = (const char * (*)(void)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_hexagon_pack_tag");
            if (!get_packed || !set_packed || !tag_fn) { fprintf(stderr, "[enc] backend has no packed-weight API\n"); return false; }
            tag = tag_fn();
            bt = ggml_backend_get_default_buffer_type(htp);
        }
        const size_t al = ggml_backend_buft_get_alignment(bt);
        // a SLIM file (slim_gguf.py): vocabulary, metadata and the token embeddings; the layers exist in the .hexpack only
        const bool slim = gguf_find_key(g, "pulsex.slim") >= 0;
        const std::string pack_path = path.substr(0, path.rfind('.')) + ".hexpack";
        if (slim) {
            if (!packed) { fprintf(stderr, "[enc] a slim file runs on the NPU only (the CPU mode reads raw tensors from the full GGUF)\n"); return false; }
            struct _stat64 st;
            if (_stat64(path.c_str(), &st) != 0) return false;
            const size_t n_want = (size_t) n_layer_run * 11;
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
                    fprintf(stderr, "[enc] %s is shorter than its own table says (%llu of %llu bytes) - a broken copy or download\n", pack_path.c_str(),
                            (unsigned long long) ps.st_size, (unsigned long long) need);
                    return false;
                }
            }
            for (int il = 0; ok && il < n_layer_run; il++) {
                size_t lb = 0;
                for (int k = 0; k < 11; k++) {
                    const PackEnt & e = ents[(size_t) il * 11 + k];
                    ok = ok && tname(il, k) == e.name;
                    lb += GGML_PAD(e.size, al);
                }
                slot_size = lb > slot_size ? lb : slot_size;
            }
            if (!ok) {
                fprintf(stderr, "[enc] %s is a slim file and the pack beside it is missing or does not fit (backend tag %s).\n"
                                "[enc] It can only be rebuilt from the full GGUF (run the encoder once on that, then slim_gguf.py).\n", path.c_str(), tag);
                return false;
            }
        }
        for (int il = 0; !slim && il < n_layer_run; il++) {
            size_t lb = 0;
            for (int k = 0; k < 11; k++) {
                const std::string nm = tname(il, k);
                int64_t id = gguf_find_tensor(g, nm.c_str());
                if (id < 0) { fprintf(stderr, "[enc] missing %s\n", nm.c_str()); return false; }
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
        }
        for (int s = 0; s < 2; s++) {
            slot[s] = ggml_backend_buft_alloc_buffer(bt, slot_size);
            if (!slot[s]) return false;
            ggml_backend_buffer_set_usage(slot[s], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        ggml_backend_t bk[2] = { htp, cpu };
        sched = packed ? ggml_backend_sched_new(bk, nullptr, 2, 2048, false, true) : ggml_backend_sched_new(&cpu, nullptr, 1, 2048, false, true);
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
        fprintf(stderr, "[enc] pack built: %zu tensors, %.1f s -> %s\n", ents.size(), now_s() - t0, path.c_str());
        return true;
    }

    ggml_context * load_layer(int il, int s, Layer & L, std::vector<uint8_t> & stage) {
        ggml_init_params p = { ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context * c = ggml_init(p);
        ggml_tallocr ta = ggml_tallocr_new(slot[s]);
        ggml_tensor ** dst[11] = { &L.an, &L.wq, &L.wk, &L.wv, &L.qn, &L.kn, &L.wo, &L.fn, &L.wg, &L.wu, &L.wd };
        for (int k = 0; k < 11; k++) {
            const PackEnt & e = ents[(size_t) il * 11 + k];
            ggml_tensor * t = ggml_new_tensor(c, (ggml_type) e.type, GGML_MAX_DIMS, e.ne);
            ggml_set_name(t, e.name);
            if (ggml_tallocr_alloc(&ta, t) != GGML_STATUS_SUCCESS) { ggml_free(c); return nullptr; }
            const double t0 = now_s();
            if (packed) {
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
            *dst[k] = t;
        }
        return c;
    }

    // tok: n positions (prompt + padding), the first n_real are the prompt. out: [n x 3*n_embd], the hidden states
    // after layers taps[0..2] side by side per token.
    bool encode(const std::vector<llama_token> & tok, int n_real, const int taps[3], std::vector<float> & out) {
        const int64_t n = (int64_t) tok.size();
        const bool log = env_int("FLUX_ENC_LOG", 0) != 0;
        std::vector<float> x((size_t) n * n_embd);
        {   // token embeddings: dequantize the rows on the host
            const int64_t id = gguf_find_tensor(g, "token_embd.weight");
            if (id < 0) return false;
            const ggml_type ty = gguf_get_tensor_type(g, id);
            const size_t rb = ggml_row_size(ty, n_embd);
            const auto * tr = ggml_get_type_traits(ty);
            std::vector<uint8_t> row(rb);
            llama_token last = -1;
            for (int64_t i = 0; i < n; i++) {
                if (tok[i] == last) { memcpy(&x[(size_t) i * n_embd], &x[(size_t) (i - 1) * n_embd], (size_t) n_embd * sizeof(float)); continue; }
                last = tok[i];
                _fseeki64(gf, (long long) (data_base + gguf_get_tensor_offset(g, id) + (size_t) tok[i] * rb), SEEK_SET);
                if (fread(row.data(), 1, rb, gf) != rb) return false;
                if (ty == GGML_TYPE_F32) memcpy(&x[(size_t) i * n_embd], row.data(), rb);
                else tr->to_float(row.data(), &x[(size_t) i * n_embd], n_embd);
            }
        }
        // mask [n_kv = n, n_q = n] f16: causal AND the key is a prompt token
        std::vector<ggml_fp16_t> mask((size_t) n * n);
        const ggml_fp16_t open = ggml_fp32_to_fp16(0.0f), shut = ggml_fp32_to_fp16(-INFINITY);
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) mask[(size_t) i * n + j] = (j <= i && j < n_real) ? open : shut;
        std::vector<int32_t> pos((size_t) n);
        for (int64_t i = 0; i < n; i++) pos[i] = (int32_t) i;

        out.assign((size_t) n * 3 * n_embd, 0.0f);
        std::vector<uint8_t> stage;
        std::vector<float> y((size_t) n * n_embd);
        for (int il = 0; il < n_layer_run; il++) {
            const int s = il & 1;
            if (lctx[s]) { ggml_free(lctx[s]); lctx[s] = nullptr; }
            Layer L{};
            lctx[s] = load_layer(il, s, L, stage);
            if (!lctx[s]) { fprintf(stderr, "[enc] layer %d failed to load\n", il); return false; }

            const double tc = now_s();
            ggml_init_params p = { ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(256, false), nullptr, true };
            ggml_context * c = ggml_init(p);
            ggml_tensor * xin = ggml_new_tensor_2d(c, GGML_TYPE_F32, n_embd, n);
            ggml_tensor * tpos = ggml_new_tensor_1d(c, GGML_TYPE_I32, n);
            ggml_tensor * tmask = ggml_new_tensor_2d(c, GGML_TYPE_F16, n, n);
            ggml_set_input(xin); ggml_set_input(tpos); ggml_set_input(tmask);

            auto norm = [&](ggml_tensor * t, ggml_tensor * w) { return ggml_mul(c, ggml_rms_norm(c, t, eps), w); };
            auto rope = [&](ggml_tensor * t) {
                return ggml_rope_ext(c, t, tpos, nullptr, (int) head_dim, GGML_ROPE_TYPE_NEOX, n_ctx_orig, rope_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
            };
            ggml_tensor * cur = norm(xin, L.an);
            ggml_tensor * q = ggml_reshape_3d(c, ggml_mul_mat(c, L.wq, cur), head_dim, n_head, n);
            ggml_tensor * k = ggml_reshape_3d(c, ggml_mul_mat(c, L.wk, cur), head_dim, n_head_kv, n);
            ggml_tensor * v = ggml_reshape_3d(c, ggml_mul_mat(c, L.wv, cur), head_dim, n_head_kv, n);
            q = rope(norm(q, L.qn));
            k = rope(norm(k, L.kn));
            q = ggml_permute(c, q, 0, 2, 1, 3);
            k = ggml_cast(c, ggml_permute(c, k, 0, 2, 1, 3), GGML_TYPE_F16);
            v = ggml_cast(c, ggml_permute(c, v, 0, 2, 1, 3), GGML_TYPE_F16);
            ggml_tensor * a = ggml_flash_attn_ext(c, q, k, v, tmask, 1.0f / sqrtf((float) head_dim), 0.0f, 0.0f);
            ggml_prec_set_acc(a, GGML_PREC_F32);
            a = ggml_reshape_2d(c, a, a->ne[0] * a->ne[1], a->ne[2] * a->ne[3]);
            ggml_tensor * h = ggml_add(c, ggml_mul_mat(c, L.wo, a), xin);
            cur = norm(h, L.fn);
            cur = ggml_swiglu_split(c, ggml_mul_mat(c, L.wg, cur), ggml_mul_mat(c, L.wu, cur));
            ggml_tensor * xo = ggml_add(c, ggml_mul_mat(c, L.wd, cur), h);
            ggml_set_output(xo);

            ggml_cgraph * gr = ggml_new_graph_custom(c, 256, false);
            ggml_build_forward_expand(gr, xo);
            ggml_backend_sched_reset(sched);
            if (!ggml_backend_sched_alloc_graph(sched, gr)) { ggml_free(c); fprintf(stderr, "[enc] alloc failed\n"); return false; }
            ggml_backend_tensor_set(xin, x.data(), 0, x.size() * sizeof(float));
            ggml_backend_tensor_set(tpos, pos.data(), 0, pos.size() * sizeof(int32_t));
            ggml_backend_tensor_set(tmask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
            if (ggml_backend_sched_graph_compute(sched, gr) != GGML_STATUS_SUCCESS) { ggml_free(c); fprintf(stderr, "[enc] compute failed at layer %d\n", il); return false; }
            ggml_backend_tensor_get(xo, y.data(), 0, y.size() * sizeof(float));
            if (il == 0 && log) fprintf(stderr, "[enc] graph: %d nodes, %d splits\n", ggml_graph_n_nodes(gr), ggml_backend_sched_get_n_splits(sched));
            ggml_free(c);
            x.swap(y);
            t_comp += now_s() - tc;
            if (log) {
                double s2 = 0;
                for (float f : x) s2 += (double) f * f;
                fprintf(stderr, "[enc] layer %2d  |h| %.3f\n", il, sqrt(s2 / (double) x.size()));
            }
            for (int t = 0; t < 3; t++)
                if (il + 1 == taps[t])
                    for (int64_t i = 0; i < n; i++)
                        memcpy(&out[(size_t) i * 3 * n_embd + (size_t) t * n_embd], &x[(size_t) i * n_embd], (size_t) n_embd * sizeof(float));
        }
        return true;
    }

    // every way out goes through the destructor: the NPU session is closed whichever return or exception ends the run
    void close() {
        for (auto & l : lctx) if (l) { ggml_free(l); l = nullptr; }
        if (sched) { ggml_backend_sched_free(sched); sched = nullptr; }
        for (auto & b : slot) if (b) { ggml_backend_buffer_free(b); b = nullptr; }
        if (htp) { ggml_backend_free(htp); htp = nullptr; }
        if (cpu) { ggml_backend_free(cpu); cpu = nullptr; }
        if (pk) { fclose(pk); pk = nullptr; }
        if (gf) { fclose(gf); gf = nullptr; }
        if (g) { gguf_free(g); g = nullptr; }
    }
    ~Engine() { close(); }
};

struct Vocab {            // llama.cpp, vocab only
    llama_model * m = nullptr;
    explicit Vocab(const std::string & path) {
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        m = llama_model_load_from_file(path.c_str(), mp);
    }
    ~Vocab() { if (m) llama_model_free(m); }
    std::vector<llama_token> tokenize(const std::string & text) const {
        const llama_vocab * v = llama_model_get_vocab(m);
        std::vector<llama_token> tok(text.size() + 16);
        const int n = llama_tokenize(v, text.c_str(), (int32_t) text.size(), tok.data(), (int32_t) tok.size(), true, true);
        tok.resize(n > 0 ? (size_t) n : 0);
        return tok;
    }
};

static void quiet_log(enum ggml_log_level level, const char * text, void *) {
    static const bool all = env_int("FLUX_ENC_LOG", 0) != 0;
    if (all || level >= GGML_LOG_LEVEL_ERROR) fputs(text, stderr);
}

static int run(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: flux-encode <model.gguf> <text file> <out.f32> [device=HTP0|CPU] [n_tokens=512]\n"); return 2; }
    const std::string model = argv[1], in_path = argv[2], out_path = argv[3], device = argc > 4 ? argv[4] : "HTP0";
    const int n_tokens = argc > 5 ? atoi(argv[5]) : 512;
    const int taps[3] = { 9, 18, 27 };
    const double t0 = now_s();
    llama_log_set(quiet_log, nullptr);
    ggml_backend_load_all();
    llama_backend_init();

    std::string text;
    { std::ifstream f(in_path, std::ios::binary); std::stringstream ss; ss << f.rdbuf(); text = ss.str(); }
    if (text.empty()) { printf("ERROR empty text %s\n", in_path.c_str()); return 1; }

    std::vector<llama_token> tok;
    int n_real = 0;
    {
        Vocab V(model);
        if (!V.m) { printf("ERROR cannot load the vocabulary of %s\n", model.c_str()); return 1; }
        tok = V.tokenize(text);
        const std::vector<llama_token> pad = V.tokenize("<|endoftext|>");
        if (tok.empty() || pad.size() != 1) { printf("ERROR tokenize gave %zu tokens (padding %zu)\n", tok.size(), pad.size()); return 1; }
        if ((int) tok.size() > n_tokens) tok.resize((size_t) n_tokens);           // truncation=True, max_length
        n_real = (int) tok.size();
        tok.resize((size_t) n_tokens, pad[0]);
    }
    if (const char * tf = getenv("FLUX_ENC_TOKENS")) {
        if (FILE * o = fopen(tf, "wb")) { fwrite(tok.data(), sizeof(llama_token), tok.size(), o); fclose(o); }
    }
    if (env_int("FLUX_ENC_TOKENS_ONLY", 0)) {          // the tokenizer check (tools/token_check_1002.py): no engine, no NPU
        printf("DONE %d\n", n_real);
        fflush(stdout);
        llama_backend_free();
        return 0;
    }

    int rc = 0;
    {
        Engine E;
        if (!E.open(model, device, taps[2])) { printf("ERROR cannot open the encoder\n"); return 1; }
        const double t_ready = now_s() - t0;
        std::vector<float> out;
        const double te = now_s();
        if (!E.encode(tok, n_real, taps, out)) { printf("ERROR encode failed\n"); rc = 1; }
        else {
            FILE * o = fopen(out_path.c_str(), "wb");
            if (!o) { printf("ERROR cannot open %s\n", out_path.c_str()); rc = 1; }
            else { fwrite(out.data(), sizeof(float), out.size(), o); fclose(o); printf("DONE %d\n", n_real); }
        }
        fprintf(stderr, "[enc] %s: %d of %d tokens | ready %.2f s | encode %.2f s = read %.2f + copy %.2f + compute %.2f | slot %.1f MB x2\n",
                device.c_str(), n_real, n_tokens, t_ready, now_s() - te, E.t_read, E.t_copy, E.t_comp, E.slot_size / 1048576.0);
    }   // the engine (and the NPU session) is closed here
    fflush(stdout);
    llama_backend_free();
    return rc;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[enc] stopped: %s\n", e.what());
        return 1;
    }
}
