// pulsex_make_util.hpp - what the "make" programs share (flux-make, zimage-make): paths and the settings file,
// --progress, starting an engine as its own process, torch.randn rebuilt, and enlarging through QuickSRNet.
// 2026-10-02: cut out of flux_make.cpp unchanged, so that zimage_make.cpp is not a copy of it.
#pragma once
#include "flux_qnn_graph.hpp"
#include "flux_image.hpp"

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
// With `on_out`, every line of the stdout is handed to on_out as it comes (it is still returned whole in `out`).
static int spawn(const std::vector<std::string> & args, const std::string & cwd, const std::string & errfile, std::string & out,
                 const std::function<void(const std::string &)> & on_err = nullptr,
                 const std::function<void(const std::string &)> & on_out = nullptr) {
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
    std::string oline;
    while (ReadFile(rd, buf, sizeof buf, &got, nullptr) && got) {
        out.append(buf, got);
        if (on_out)                                // the program's stdout, line by line, as it comes
            for (DWORD i = 0; i < got; i++) {
                if (buf[i] == '\n') { on_out(oline); oline.clear(); }
                else if (buf[i] != '\r') oline += buf[i];
            }
    }
    if (on_out && !oline.empty()) on_out(oline);
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
