// flux-make: a prompt in, a picture out - the whole FLUX.2 klein chain WITHOUT Python.
//
// 2026-10-02 (for an application that people without technical knowledge can use). flux.py is the
// reference; this is the same chain in one C++ program, so an application has nothing to install beside it:
//   1. the prompt in the Qwen3 chat template                               -> flux-encode.exe   (NPU, own process)
//   2. the start noise: torch.randn's generator (mt19937, 24-bit uniforms, Box-Muller over blocks of 16), the
//      shifted sigma schedule                                              -> flux-stream.exe   (NPU, own process)
//   3. the latents de-normalised and un-patchified, the VAE decoder as the QNN context binary ONNX Runtime compiled
//      (flux_qnn_graph.hpp, QnnHtp.dll alone)                              -> in this process
//   4. the picture written as PNG by Windows' own codec (WIC)
// The engines stay separate processes on purpose: each opens the NPU, does one thing, and closes it when it exits.
// Settings come from flux.json (the file flux.py reads): which transformer / text encoder, where the engines are.
//   flux-make "prompt" [--size 512|WxH] [--seed N] [--steps 4] [--model q4|q8] [--out file.png] [--config flux.json] [--keep]
//   flux-make --prompt-file prompt.txt --progress ...     (what the window, flux_gui.cpp, runs)
//   flux-make ... --title "Summer Sale" --sub "Friday 6 pm" [--top]        a poster: text on the finished picture
//   flux-make --add-text picture.png --title ... [--sub ...] [--top] [--out file.png]     the text on an existing picture
//   flux-make --enlarge picture.png [--up N] [--title ...] [--out file.png]               an existing picture enlarged
//   --up N: the enlarged picture's short side, 1024 or 2048 (512x768 --up 1024 -> 1024x1536); the text is put on after it
//   --out name.jpg [--quality 95]: a JPEG (no colour subsampling when there is text on the picture); name.png: a PNG
// stdout: the picture's path and the three timings - the lines flux.py prints. With --progress also "PROGRESS <stage>".

#include "flux_qnn_graph.hpp"
#include "flux_image.hpp"
#include "flux_text.hpp"
#include "flux_enrich.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <thread>

static double now_s() { return fluxqnn::now_ms() / 1000.0; }

static std::wstring wide(const std::string & s) {
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), (int) s.size(), &w[0], n);
    return w;
}

// the string value of "key" in a flat JSON text (flux.json: one object of strings inside another); "" when absent
static std::string json_str(const std::string & js, const std::string & key) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    k = js.find(':', k);
    const size_t a = js.find('"', k), b = js.find('"', a + 1);
    return a == std::string::npos || b == std::string::npos ? "" : js.substr(a + 1, b - a - 1);
}
static std::string winpath(std::string p) { std::replace(p.begin(), p.end(), '/', '\\'); return p; }
static bool exists(const std::string & p) { return GetFileAttributesW(wide(p).c_str()) != INVALID_FILE_ATTRIBUTES; }
static std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}
static std::string full_path(const std::string & p) {          // absolute, ".." folded
    wchar_t b[2048];
    const DWORD n = GetFullPathNameW(wide(p).c_str(), 2048, b, nullptr);
    return n && n < 2048 ? narrow(b) : p;
}
static std::string dir_of(const std::string & p) { const size_t s = p.find_last_of("\\/"); return s == std::string::npos ? std::string(".") : p.substr(0, s); }
static std::string exe_dir() { wchar_t b[2048] = { 0 }; GetModuleFileNameW(nullptr, b, 2048); return dir_of(narrow(b)); }
// a path from flux.json: an absolute one as it is, a relative one from the folder flux.json is in; "" when the key is absent
static std::string cfg(const std::string & js, const std::string & base, const std::string & key) {
    const std::string p = winpath(json_str(js, key));
    if (p.empty()) return p;
    const bool absolute = (p.size() > 1 && p[1] == ':') || p[0] == '\\';
    return full_path(absolute ? p : base + "\\" + p);
}

// --progress: one line per stage on stdout ("PROGRESS text", "PROGRESS step 2 4", ...) for a window that shows a bar
static bool g_progress = false;
static void progress(const std::string & what) {
    if (!g_progress) return;
    printf("PROGRESS %s\n", what.c_str());
    fflush(stdout);
}

// run a program in `cwd`, its stderr into `errfile`, its stdout returned; the exit code.
// With `on_err`, the stderr goes through a pipe instead: every line is handed to on_err and written to `errfile`.
static int spawn(const std::vector<std::string> & args, const std::string & cwd, const std::string & errfile, std::string & out,
                 const std::function<void(const std::string &)> & on_err = nullptr) {
    std::wstring cmd;
    for (const auto & a : args) cmd += (cmd.empty() ? L"\"" : L" \"") + wide(a) + L"\"";
    SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr, erd = nullptr, ewr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) throw std::runtime_error("CreatePipe failed");
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE ef = CreateFileW(wide(errfile).c_str(), GENERIC_WRITE, FILE_SHARE_READ, on_err ? nullptr : &sa, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (on_err) {
        if (!CreatePipe(&erd, &ewr, &sa, 0)) throw std::runtime_error("CreatePipe failed");
        SetHandleInformation(erd, HANDLE_FLAG_INHERIT, 0);
    }
    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr; si.hStdError = on_err ? ewr : ef; si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION pi{};
    const BOOL ok = CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, wide(cwd).c_str(), &si, &pi);
    CloseHandle(wr);
    if (ewr) CloseHandle(ewr);
    std::thread err_reader;
    if (ok && on_err)
        err_reader = std::thread([&]() {              // the engine's log: to the file as before, and line by line to on_err
            std::string line;
            char b[2048];
            DWORD n = 0, wrote = 0;
            while (ReadFile(erd, b, sizeof b, &n, nullptr) && n) {
                if (ef != INVALID_HANDLE_VALUE) WriteFile(ef, b, n, &wrote, nullptr);
                for (DWORD i = 0; i < n; i++) {
                    if (b[i] == '\n') { on_err(line); line.clear(); }
                    else if (b[i] != '\r') line += b[i];
                }
            }
            if (!line.empty()) on_err(line);
        });
    if (!ok) {
        CloseHandle(rd);
        if (erd) CloseHandle(erd);
        if (ef != INVALID_HANDLE_VALUE) CloseHandle(ef);
        throw std::runtime_error("cannot start " + args[0]);
    }
    out.clear();
    char buf[4096];
    DWORD got = 0;
    while (ReadFile(rd, buf, sizeof buf, &got, nullptr) && got) out.append(buf, got);
    CloseHandle(rd);
    if (err_reader.joinable()) err_reader.join();
    if (erd) CloseHandle(erd);
    if (ef != INVALID_HANDLE_VALUE) CloseHandle(ef);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD rc = 1;
    GetExitCodeProcess(pi.hProcess, &rc);
    CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
    return (int) rc;
}

static std::string tail_of(const std::string & file, size_t n = 1200) {
    std::vector<uint8_t> v;
    try { v = fluxqnn::read_file(file); } catch (...) { return ""; }
    return std::string(v.begin() + (v.size() > n ? v.size() - n : 0), v.end());
}

// torch.randn(n) on the CPU for a seeded generator, n >= 16 (torch_noise.py is the same thing in numpy)
static std::vector<float> torch_randn(size_t n, uint32_t seed) {
    std::mt19937 g(seed);
    const size_t extra = n % 16 ? 16 : 0;
    std::vector<float> u(n + extra);
    for (auto & v : u) v = (float) (g() & 0xFFFFFFu) * (1.0f / 16777216.0f);
    auto fill16 = [](float * d) {
        for (int j = 0; j < 8; j++) {
            const float u1 = 1.0f - d[j], u2 = d[j + 8];
            const float radius = sqrtf(-2.0f * logf(u1));
            const float theta = (float) (2.0 * 3.14159265358979323846 * (double) u2);
            d[j] = radius * cosf(theta);
            d[j + 8] = radius * sinf(theta);
        }
    };
    for (size_t i = 0; i + 16 <= n; i += 16) fill16(&u[i]);
    if (extra) { float t[16]; memcpy(t, &u[n], sizeof t); fill16(t); memcpy(&u[n - 16], t, sizeof t); }
    u.resize(n);
    return u;
}

// FlowMatchEuler with dynamic (exponential) shifting, as the FLUX.2 klein pipeline sets it up; the last sigma is 0
static std::vector<double> sigmas_for(int n_tokens, int steps) {
    const double a1 = 8.73809524e-05, b1 = 1.89833333, a2 = 0.00016927, b2 = 0.45666666;
    double mu;
    if (n_tokens > 4300) mu = a2 * n_tokens + b2;
    else {
        const double m200 = a2 * n_tokens + b2, m10 = a1 * n_tokens + b1, k = (m200 - m10) / 190.0;
        mu = k * steps + (m200 - 200.0 * k);
    }
    std::vector<double> s;
    for (int i = 0; i < steps; i++) {
        const double v = steps == 1 ? 1.0 : 1.0 + (1.0 / steps - 1.0) * i / (steps - 1);     // linspace(1, 1/steps, steps)
        s.push_back(exp(mu) / (exp(mu) + (1.0 / v - 1.0)));
    }
    s.push_back(0.0);
    return s;
}

static void write_text(const std::string & p, const std::string & s) {
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + p);
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
}

// Enlarge an RGB picture so that its SHORT side becomes `up`: QuickSRNet x4 (the context binary `sr`, a graph for
// 512 x 512) on the session S, a Lanczos to the target, mixed 0.6 with a Lanczos of the picture itself.
static void enlarge(pcore_npu::QnnHtpSession & S, const std::string & sr, std::vector<uint8_t> & rgb, int & w, int & h, int up) {
    fluxqnn::Graph U;
    U.load(S, sr);
    const size_t tp = (size_t) 512 * 512, bp = (size_t) 2048 * 2048;
    if (U.in.count != tp * 3 || U.out.count != bp * 3) throw std::runtime_error("the upscaler binary is not 512 -> 2048: " + U.in.str() + " -> " + U.out.str());
    // The graph takes 512 x 512. A larger picture goes through it as tiles that overlap by a quarter at least;
    // each tile counts most in its middle and least at its rim (a triangle each way), so the seams cross-fade.
    // A 512 x 512 picture is one tile and is copied as it comes - the same pixels as before there were tiles.
    auto places = [](int L) {
        std::vector<int> v;
        if (L <= 512) { v.push_back(0); return v; }
        const int n = (int) ceil((L - 512) / 384.0) + 1;
        for (int i = 0; i < n; i++) v.push_back((int) llround((double) i * (L - 512) / (n - 1)));
        return v;
    };
    const std::vector<int> xs = places(w), ys = places(h);
    const bool one = xs.size() == 1 && ys.size() == 1;
    fluximg::Rgb x4, base;        // (`small` is a macro in the Windows headers)
    x4.w = 4 * w; x4.h = 4 * h;
    x4.px.resize((size_t) x4.w * x4.h * 3);
    std::vector<float> acc, wsum;
    if (!one) { acc.assign(x4.px.size(), 0.0f); wsum.assign((size_t) x4.w * x4.h, 0.0f); }
    std::vector<uint8_t> nchw(tp * 3);
    for (int ty : ys)
        for (int tx : xs) {
            for (int y = 0; y < 512; y++)
                for (int x = 0; x < 512; x++) {
                    const uint8_t * s = &rgb[((size_t) (ty + y) * w + tx + x) * 3];
                    for (int c = 0; c < 3; c++) nchw[(size_t) c * tp + (size_t) y * 512 + x] = s[c];
                }
            const std::vector<uint8_t> big = U.run_u8(nchw.data());            // [3, 2048, 2048]
            for (int y = 0; y < 2048; y++) {
                const float wy = (float) (std::min)(y + 1, 2048 - y) / 1024.0f;
                const size_t row = (size_t) (4 * ty + y) * x4.w + (size_t) 4 * tx;
                for (int x = 0; x < 2048; x++) {
                    const size_t o = row + x, i = (size_t) y * 2048 + x;
                    if (one) { for (int c = 0; c < 3; c++) x4.px[o * 3 + c] = big[(size_t) c * bp + i]; continue; }
                    const float wt = wy * (float) (std::min)(x + 1, 2048 - x) / 1024.0f;
                    for (int c = 0; c < 3; c++) acc[o * 3 + c] += wt * (float) big[(size_t) c * bp + i];
                    wsum[o] += wt;
                }
            }
        }
    if (!one)
        for (size_t o = 0; o < wsum.size(); o++)
            for (int c = 0; c < 3; c++) {
                const float v = nearbyintf(acc[o * 3 + c] / wsum[o]);
                x4.px[o * 3 + c] = (uint8_t) (v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v);
            }
    acc = std::vector<float>(); wsum = std::vector<float>();
    base.w = w; base.h = h; base.px = rgb;
    const double k = (double) up / (std::min)(w, h);                           // --up is the short side
    const int W2 = (int) llround(w * k), H2 = (int) llround(h * k);
    const fluximg::Rgb sharp = fluximg::resize_lanczos(x4, W2, H2), soft = fluximg::resize_lanczos(base, W2, H2);
    rgb.resize(sharp.px.size());
    for (size_t i = 0; i < rgb.size(); i++) {
        const float v = nearbyintf(0.6f * (float) sharp.px[i] + 0.4f * (float) soft.px[i]);
        rgb[i] = (uint8_t) (v < 0.0f ? 0.0f : v > 255.0f ? 255.0f : v);
    }
    w = W2; h = H2;
}

static int run(int argc, char ** argv) {
    std::string prompt, size = "512", model, out, cfg_path = exe_dir() + "\\flux.json", noise_only, title, subtitle, add_text, enlarge_src, enrich_mode = "off";
    bool text_top = false, enrich_only = false;
    std::vector<std::string> refs;
    int tokens = 0, up = 0, quality = 95;
    uint32_t seed = 0;
    int steps = 4, n_text = 512;
    bool keep = false;
    if (const char * e = getenv("FLUX_CONFIG")) if (*e) cfg_path = e;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { if (i + 1 >= argc) throw std::runtime_error(a + " needs a value"); return argv[++i]; };
        if (a == "--size") size = val();
        else if (a == "--seed") seed = (uint32_t) strtoul(val().c_str(), nullptr, 10);
        else if (a == "--steps") steps = atoi(val().c_str());
        else if (a == "--model") model = val();
        else if (a == "--out") out = val();
        else if (a == "--config") cfg_path = val();
        else if (a == "--text-tokens") n_text = atoi(val().c_str());
        else if (a == "--keep") keep = true;
        else if (a == "--image") refs.push_back(val());
        else if (a == "--up") up = atoi(val().c_str());
        else if (a == "--noise-only") noise_only = val();
        else if (a == "--tokens") tokens = atoi(val().c_str());
        else if (a == "--progress") g_progress = true;
        else if (a == "--title") title = val();                 // text on the finished picture (flux_text.hpp): posters
        else if (a == "--sub" || a == "--subtitle") subtitle = val();
        else if (a == "--top") text_top = true;
        else if (a == "--quality") quality = atoi(val().c_str());   // of a JPEG (the output name ends in .jpg), 1..100
        else if (a == "--add-text") add_text = val();           // no picture is made: the text goes on this existing picture
        else if (a == "--enlarge") enlarge_src = val();         // no picture is made: this existing picture is enlarged (--up N = its new short side)
        else if (a == "--enrich") enrich_mode = val();          // nouns | full | off - the Norrland theme (flux_enrich.hpp)
        else if (a == "--enrich-only") enrich_only = true;      // print the prompt that would be sent, and stop
        else if (a == "--prompt-file") {           // the prompt as a UTF-8 file: any quote, backslash or line break survives
            const std::vector<uint8_t> v = fluxqnn::read_file(val());
            prompt.assign(v.begin(), v.end());
            if (prompt.compare(0, 3, "\xEF\xBB\xBF") == 0) prompt.erase(0, 3);
            while (!prompt.empty() && (prompt.back() == '\n' || prompt.back() == '\r' || prompt.back() == ' ')) prompt.pop_back();
        }
        else if (prompt.empty()) prompt = a;
        else throw std::runtime_error("what is " + a + "?");
    }
    if (!noise_only.empty()) {             // the start noise alone, packed [tokens, 128] - flux.py takes its noise from here
        if (tokens < 1) throw std::runtime_error("--noise-only needs --tokens");
        const std::vector<float> u = torch_randn((size_t) 128 * tokens, seed);
        std::vector<float> packed((size_t) tokens * 128);
        for (int c = 0; c < 128; c++)
            for (int k = 0; k < tokens; k++) packed[(size_t) k * 128 + c] = u[(size_t) c * tokens + k];
        FILE * f = fopen(noise_only.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + noise_only);
        fwrite(packed.data(), 4, packed.size(), f);
        fclose(f);
        return 0;
    }
    if (!add_text.empty()) {               // the text alone, on a picture that exists (any format Windows reads) - no NPU
        if (title.empty() && subtitle.empty()) throw std::runtime_error("--add-text needs --title and / or --sub");
        fluximg::Rgb im = fluximg::load(add_text);
        fluxtext::add_text(im, title, subtitle, text_top);
        if (out.empty()) { const size_t d = add_text.find_last_of('.'); out = (d == std::string::npos ? add_text : add_text.substr(0, d)) + "_text.png"; }
        fluximg::save(out, im, quality, true);                  // text on it: full colour resolution in a JPEG
        printf("%s  %dx%d, text added\n", out.c_str(), im.w, im.h);
        fflush(stdout);
        return 0;
    }
    if (prompt.empty() && enlarge_src.empty()) { fprintf(stderr, "usage: flux-make \"prompt\" [--size 512|WxH] [--seed N] [--steps 4] [--model q4|q8] [--up N] [--title text] [--sub text] [--top] [--out file.png]\n"); return 2; }
    int w = atoi(size.c_str()), h = w;
    { const size_t x = size.find_first_of("xX"); if (x != std::string::npos) { w = atoi(size.substr(0, x).c_str()); h = atoi(size.substr(x + 1).c_str()); } }
    if (w < 64 || h < 64 || w % 16 || h % 16) throw std::runtime_error("--size: width and height must be multiples of 16");
    const int hp = h / 16, wp = w / 16, T = hp * wp;

    const std::vector<uint8_t> cj = fluxqnn::read_file(cfg_path);
    const std::string js(cj.begin(), cj.end());
    const std::string base = dir_of(full_path(cfg_path));        // relative paths in flux.json start here
    if (model.empty()) model = json_str(js, "model");
    if (model != "q4" && model != "q8") model = "q4";
    // the Norrland theme: Swedish words get their English nouns in front (and, "full", material words after)
    std::string sent = prompt;
    if (enrich_mode == "nouns" || enrich_mode == "full") {
        const std::string cues = cfg(js, base, "cues");
        const fluxenrich::Table tab = fluxenrich::load(cues);
        if (!tab.ok) fprintf(stderr, "(--enrich: no cue table at %s - the prompt is sent as typed)\n", cues.c_str());
        else sent = fluxenrich::enrich(tab, prompt, enrich_mode == "full");
    } else if (enrich_mode != "off") throw std::runtime_error("--enrich takes nouns, full or off");
    if (enrich_only) { printf("%s\n", sent.c_str()); return 0; }
    const std::string bin = json_str(js, "bin").empty() ? exe_dir() : cfg(js, base, "bin"), dit = cfg(js, base, "transformer_" + model),
                      enc = cfg(js, base, "encoder_" + model), vq = cfg(js, base, "vae_qnn"), rt = cfg(js, base, "qnn_runtime");
    const std::string wxh = std::to_string(w) + "x" + std::to_string(h);
    const std::string dec = vq + "\\flux2-vae-decoder-" + wxh + "_ctx_qnn.bin", encq = vq + "\\flux2-vae-encoder-" + wxh + "_ctx_qnn.bin";
    std::string sr = cfg(js, base, "quicksrnet_x4");
    if (sr.size() > 5 && sr.substr(sr.size() - 5) == ".onnx") sr = sr.substr(0, sr.size() - 5) + "_ctx_qnn.bin";
    if (!enlarge_src.empty()) {            // an EXISTING picture through the upscaler (and the text, if any): the NPU for a second, no model
        fluximg::Rgb im = fluximg::load(enlarge_src);
        const int m = (std::min)(im.w, im.h);
        if (!up) up = m < 1024 ? 2 * m : 2048;
        if (m < 512 || (std::max)(im.w, im.h) > 1536 || up <= m || up > 4 * m || !exists(sr) || !exists(rt))
            throw std::runtime_error("--enlarge: the picture's short side must be 512 or more, its long side 1536 or less, --up between them and four times the short side, and the QuickSRNet context must exist (" + sr + ")");
        const double t0 = now_s();
        SetDllDirectoryW(wide(rt).c_str());
        {
            pcore_npu::QnnHtpSession S;                // closed when the picture is enlarged
            if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
            progress("upscale");
            enlarge(S, sr, im.px, im.w, im.h, up);
        }
        fluxtext::add_text(im, title, subtitle, text_top);
        if (out.empty()) { const size_t d = enlarge_src.find_last_of('.'); out = (d == std::string::npos ? enlarge_src : enlarge_src.substr(0, d)) + "_" + std::to_string(up) + ".png"; }
        fluximg::save(out, im, quality, !title.empty() || !subtitle.empty());
        printf("%s  %dx%d, enlarged in %.1f s\n", out.c_str(), im.w, im.h, now_s() - t0);
        fflush(stdout);
        return 0;
    }
    if (!refs.empty() && !exists(encq)) throw std::runtime_error("no compiled picture encoder for " + wxh + " (" + encq +
                                                                 ") - edit one picture of that size with flux.py first, it exports and compiles it");
    if (up && up != 1024 && up != 2048) throw std::runtime_error("--up takes 1024 or 2048");
    // --up N: N is the enlarged picture's SHORT side (512x768 --up 1024 -> 1024x1536). The upscaler is a x4 graph for
    // 512 x 512, run on overlapping tiles, so any picture of at least 512 can be enlarged up to four times.
    if (up && ((std::min)(w, h) < 512 || up <= (std::min)(w, h) || up > 4 * (std::min)(w, h) || !exists(sr))) {
        fprintf(stderr, "(--up %d: the short side must be at least 512 and grow at most four times, and the QuickSRNet context %s must exist - keeping %s)\n", up, sr.c_str(), wxh.c_str());
        up = 0;
    }
    for (const std::string * f : { &bin, &dit, &enc, &rt })
        if (f->empty() || !exists(*f)) throw std::runtime_error("flux.json: missing for model " + model + ": " + *f);
    if (!exists(dec)) throw std::runtime_error("no compiled picture decoder for " + std::to_string(w) + "x" + std::to_string(h) + " (" + dec +
                                               ") - make one picture of that size with flux.py first, it exports and compiles it");

    const double T0 = now_s();
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    const std::string wd = std::string(tmp) + "pulsex_flux_" + std::to_string(GetCurrentProcessId()) + "_" + std::to_string(GetTickCount64() & 0xFFFFFF);
    CreateDirectoryW(wide(wd).c_str(), nullptr);
    for (const char * k : { "FLUX_TAP", "FLUX_DUMP", "FLUX_BLOCKS", "FLUX_STEPS" }) SetEnvironmentVariableA(k, nullptr);

    // 1. text
    if (sent != prompt) { printf("sent: %s\n", sent.c_str()); fflush(stdout); }      // never hidden: what the model is given
    write_text(wd + "\\prompt.txt", "<|im_start|>user\n" + sent + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    std::string so;
    double t = now_s();
    progress("text");
    if (spawn({ bin + "\\flux-encode.exe", enc, wd + "\\prompt.txt", wd + "\\text.f32", "HTP0", std::to_string(n_text) }, bin, wd + "\\encode.log", so) != 0 ||
        so.find("DONE") == std::string::npos)
        throw std::runtime_error("the text encoder failed: " + so + tail_of(wd + "\\encode.log"));
    const int n_real = atoi(so.substr(so.find("DONE") + 4).c_str());
    const double t_enc = now_s() - t;

    // reference pictures (editing): cover + crop to the picture's size, the VAE encoder, patchify, normalise, pack
    double t_ref = 0;
    SetDllDirectoryW(wide(rt).c_str());
    const std::vector<uint8_t> braw = fluxqnn::read_file(vq + "\\bn.f32");                 // mean[128], var[128], eps
    if (braw.size() != 257 * 4) throw std::runtime_error("bn.f32 has the wrong size");
    const float * bn = (const float *) braw.data();
    if (!refs.empty()) {
        t = now_s();
        progress("reference");
        pcore_npu::QnnHtpSession S;                    // gone before flux-stream opens the NPU
        if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
        fluxqnn::Graph E;
        E.load(S, encq);
        const size_t plane = (size_t) w * h;
        if (E.in.count != plane * 3 || E.out.count != (size_t) 32 * 4 * T) throw std::runtime_error("the encoder binary is for another size: " + E.in.str() + " -> " + E.out.str());
        FILE * f = fopen((wd + "\\ref.f32").c_str(), "wb");
        std::string meta;
        for (const auto & file : refs) {
            const fluximg::Rgb im = fluximg::cover(fluximg::load(file), w, h);
            std::vector<float> x(plane * 3);
            for (size_t q = 0; q < plane; q++)
                for (int c = 0; c < 3; c++) x[(size_t) c * plane + q] = (float) im.px[q * 3 + c] / 127.5f - 1.0f;
            const std::vector<float> z = E.run(x.data());                                   // [32, 2hp, 2wp]
            std::vector<float> packed((size_t) T * 128);
            const int LH = 2 * hp, LW = 2 * wp;
            for (int ty = 0; ty < hp; ty++)
                for (int tx = 0; tx < wp; tx++)
                    for (int ch = 0; ch < 128; ch++) {
                        const int c = ch / 4, i = (ch / 2) % 2, j = ch % 2;
                        const float v = z[((size_t) c * LH + (2 * ty + i)) * LW + (2 * tx + j)];
                        if (!std::isfinite(v)) throw std::runtime_error("the encoder returned non-finite values for " + file);
                        packed[((size_t) ty * wp + tx) * 128 + ch] = (v - bn[ch]) / sqrtf(bn[128 + ch] + bn[256]);
                    }
            fwrite(packed.data(), 4, packed.size(), f);
            meta += std::to_string(hp) + " " + std::to_string(wp) + "\n";
        }
        fclose(f);
        write_text(wd + "\\ref_meta.txt", meta);
        t_ref = now_s() - t;
    }

    // 2. transformer
    {
        const std::vector<float> u = torch_randn((size_t) 128 * T, seed);            // = torch.randn((1, 128, hp, wp))
        std::vector<float> packed((size_t) T * 128);
        for (int c = 0; c < 128; c++)
            for (int k = 0; k < T; k++) packed[(size_t) k * 128 + c] = u[(size_t) c * T + k];
        FILE * f = fopen((wd + "\\noise.f32").c_str(), "wb");
        fwrite(packed.data(), 4, packed.size(), f);
        fclose(f);
        write_text(wd + "\\meta.txt", std::to_string(hp) + " " + std::to_string(wp) + " " + std::to_string(n_text) + "\n");
        std::string sg;
        char b[40];
        for (double s : sigmas_for(T, steps)) { snprintf(b, sizeof b, "%.9g ", s); sg += b; }
        write_text(wd + "\\sigmas.txt", sg + "\n");
    }
    t = now_s();
    progress("step 0 " + std::to_string(steps));
    std::function<void(const std::string &)> on_step;
    if (g_progress)
        on_step = [steps](const std::string & line) {           // "[flux] steg 2 sigma ..." = the third step is done
            if (line.compare(0, 12, "[flux] steg ") == 0) progress("step " + std::to_string(atoi(line.c_str() + 12) + 1) + " " + std::to_string(steps));
        };
    if (spawn({ bin + "\\flux-stream.exe", dit, wd, "HTP0" }, bin, wd + "\\stream.log", so, on_step) != 0)
        throw std::runtime_error("the transformer failed: " + so + tail_of(wd + "\\stream.log"));
    const double t_dit = now_s() - t;

    // 3. picture: de-normalise, un-patchify, the decoder as a QNN context
    t = now_s();
    progress("picture");
    double t_up = 0;
    std::vector<uint8_t> rgb((size_t) w * h * 3);
    {
        const std::vector<uint8_t> lraw = fluxqnn::read_file(wd + "\\lat_out.f32");
        if (lraw.size() != (size_t) T * 128 * 4) throw std::runtime_error("lat_out.f32 has the wrong size");
        const float * lat = (const float *) lraw.data();
        const int LH = 2 * hp, LW = 2 * wp;
        std::vector<float> z((size_t) 32 * LH * LW);
        for (int ty = 0; ty < hp; ty++)
            for (int tx = 0; tx < wp; tx++)
                for (int ch = 0; ch < 128; ch++) {
                    const float v = lat[((size_t) ty * wp + tx) * 128 + ch];
                    if (!std::isfinite(v)) throw std::runtime_error("the transformer returned non-finite latents");
                    const int c = ch / 4, i = (ch / 2) % 2, j = ch % 2;
                    z[((size_t) c * LH + (2 * ty + i)) * LW + (2 * tx + j)] = v * sqrtf(bn[128 + ch] + bn[256]) + bn[ch];
                }
        pcore_npu::QnnHtpSession S;                    // closed after the graphs
        if (!S.open()) throw std::runtime_error("QNN session: " + S.error());
        fluxqnn::Graph G;
        G.load(S, dec);
        if (G.in.count != z.size() || G.out.count != rgb.size()) throw std::runtime_error("the decoder binary is for another size: " + G.in.str() + " -> " + G.out.str());
        const std::vector<float> img = G.run(z.data());                                    // [3, h, w], about -1..1
        const size_t plane = (size_t) w * h;
        for (size_t p = 0; p < plane; p++)
            for (int c = 0; c < 3; c++) {
                float v = img[(size_t) c * plane + p] / 2.0f + 0.5f;
                v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
                rgb[p * 3 + c] = (uint8_t) nearbyintf(v * 255.0f);
            }
        if (up) {                                      // QuickSRNet x4 on the same session, then to the target and the 0.6 mix
            const double tu = now_s();
            progress("upscale");
            enlarge(S, sr, rgb, w, h, up);
            t_up = now_s() - tu;
        }
    }
    if (out.empty()) {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char b[80];
        snprintf(b, sizeof b, "flux_%04d%02d%02d_%02d%02d%02d_s%u.png", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, seed);
        out = b;
    }
    {
        fluximg::Rgb im;
        im.w = w; im.h = h; im.px.swap(rgb);
        fluxtext::add_text(im, title, subtitle, text_top);          // nothing when both are empty
        fluximg::save(out, im, quality, !title.empty() || !subtitle.empty());         // .jpg -> JPEG (4:4:4 with text), else PNG
    }
    const double t_vae = now_s() - t - t_up;

    printf("%s  %dx%d, %d steps, seed %u, %d prompt tokens, transformer %s_0, text encoder %s_0\n", out.c_str(), w, h, steps, seed, n_real,
           model == "q8" ? "Q8" : "Q4", model == "q8" ? "Q8" : "Q4");
    if (t_ref > 0) printf("%zu reference picture(s) encoded in %.1f s\n", refs.size(), t_ref);
    if (t_up > 0) printf("text %.1f s + transformer %.1f s + picture %.1f s (NPU) + upscale %.1f s = %.1f s\n", t_enc, t_dit, t_vae, t_up, now_s() - T0);
    else printf("text %.1f s + transformer %.1f s + picture %.1f s (NPU) = %.1f s\n", t_enc, t_dit, t_vae, now_s() - T0);
    if (keep) printf("work folder kept: %s\n", wd.c_str());
    else {
        for (const char * f : { "prompt.txt", "text.f32", "noise.f32", "meta.txt", "sigmas.txt", "lat_out.f32", "encode.log", "stream.log", "ref.f32", "ref_meta.txt" })
            DeleteFileW(wide(wd + "\\" + f).c_str());
        RemoveDirectoryW(wide(wd).c_str());
    }
    fflush(stdout);
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "flux-make: %s\n", e.what());
        return 1;
    }
}
