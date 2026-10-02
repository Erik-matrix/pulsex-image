// flux_gui.cpp - PulseX Image: the window. Dear ImGui + DX11, from the PulseX CV window (pulse_cv_gui.cpp).
//
// 2026-10-02 (a good-looking application for people without technical knowledge). The window
// does no model work itself: it runs flux-make.exe (next to this exe) with --progress and shows what comes back.
// So the NPU is opened and closed by the engines, one at a time, exactly as on the command line - and closing the
// window never kills an engine in the middle of a picture (the window waits for the picture, then goes away).
//   * what can be chosen is what EXISTS: picture sizes are the compiled decoders in vae-qnn, the quality choices are
//     the model files flux.json names, enlarging needs the QuickSRNet context, editing needs the size's encoder
//   * pictures land in Pictures\PulseX, never overwritten; what was written to get each one is kept in
//     %LOCALAPPDATA%\PulseX Image\history.tsv, so an old picture can give its description back
//   * test hooks: --shot file.png (draw a few frames, save the window, exit), --demo-run "prompt" (press the button),
//     --lang sv|en
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_dx11.h"
#include <d3d11.h>
#include <windows.h>
#include <windowsx.h>
#include <commdlg.h>
#include <shellapi.h>
#include <shlobj.h>
#include <dwmapi.h>
#include <tlhelp32.h>
#pragma comment(lib,"shell32.lib")
#pragma comment(lib,"ole32.lib")
#pragma comment(lib,"dwmapi.lib")
#pragma comment(lib,"comdlg32.lib")
#include "flux_image.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <random>
#include <thread>

// ── small things. The manifest makes the ANSI code page UTF-8, so every narrow string here is UTF-8. ──
static std::string narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, '\0');
    if (n > 0) WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}
static std::wstring to_w(const std::string & s) { return fluximg::wide(s); }
static double now_s() { return (double) GetTickCount64() / 1000.0; }
static bool file_exists(const std::string & p) { const DWORD a = GetFileAttributesW(to_w(p).c_str()); return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY); }
static std::string app_dir() {
    wchar_t b[MAX_PATH] = { 0 };
    GetModuleFileNameW(nullptr, b, MAX_PATH);
    const std::string p = narrow(b);
    const size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? std::string(".") : p.substr(0, s);
}
static std::string known_folder(REFKNOWNFOLDERID id) {
    PWSTR wp = nullptr;
    std::string out;
    if (SUCCEEDED(SHGetKnownFolderPath(id, 0, nullptr, &wp)) && wp) out = narrow(wp);
    if (wp) CoTaskMemFree(wp);
    return out;
}
static std::string read_all(const std::string & p) {
    FILE * f = _wfopen(to_w(p).c_str(), L"rb");
    if (!f) return "";
    std::string s;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
    fclose(f);
    return s;
}
static bool write_all(const std::string & p, const std::string & s, const wchar_t * mode = L"wb") {
    FILE * f = _wfopen(to_w(p).c_str(), mode);
    if (!f) return false;
    fwrite(s.data(), 1, s.size(), f);
    fclose(f);
    return true;
}
// one line of text: a line break becomes \n, a backslash \\, a tab a space
static std::string esc(const std::string & s) {
    std::string o;
    for (char c : s) {
        if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\t') o += ' ';
        else if (c != '\r') o += c;
    }
    return o;
}
static std::string unesc(const std::string & s) {
    std::string o;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '\\' && i + 1 < s.size()) { i++; o += s[i] == 'n' ? '\n' : s[i]; }
        else o += s[i];
    }
    return o;
}
static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> v;
    size_t a = 0;
    for (;;) {
        const size_t b = s.find(sep, a);
        v.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return v;
}
// the string value of "key" in flux.json (flux-make reads it the same way)
static std::string json_str(const std::string & js, const std::string & key) {
    size_t k = js.find("\"" + key + "\"");
    if (k == std::string::npos) return "";
    k = js.find(':', k);
    const size_t a = js.find('"', k), b = js.find('"', a + 1);
    if (a == std::string::npos || b == std::string::npos) return "";
    std::string v = js.substr(a + 1, b - a - 1);
    std::replace(v.begin(), v.end(), '/', '\\');
    return v;
}
static std::string name_of(const std::string & p) { const size_t s = p.find_last_of("\\/"); return s == std::string::npos ? p : p.substr(s + 1); }
static bool is_picture_file(const std::string & p) {
    const size_t d = p.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = p.substr(d);
    for (char & c : e) c = (char) tolower((unsigned char) c);
    for (const char * k : { ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif" }) if (e == k) return true;
    return false;
}

// ── language: English, with Swedish behind the button in the title bar ──
static int g_lang = 1;                       // 0 = svenska, 1 = English
static const char * tr(const char * sv, const char * en) { return g_lang ? en : sv; }

// ── DX11 plumbing (trimmed from the official ImGui example) ──
static ID3D11Device *           g_pd3dDevice        = nullptr;
static ID3D11DeviceContext *    g_pd3dDeviceContext = nullptr;
static IDXGISwapChain *         g_pSwapChain        = nullptr;
static ID3D11RenderTargetView * g_mainRTV           = nullptr;
static void CreateRenderTarget() {
    ID3D11Texture2D * back = nullptr;
    g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (back) { g_pd3dDevice->CreateRenderTargetView(back, nullptr, &g_mainRTV); back->Release(); }
}
static void CleanupRenderTarget() { if (g_mainRTV) { g_mainRTV->Release(); g_mainRTV = nullptr; } }
static bool CreateDeviceD3D(HWND hWnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2; sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60; sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT; sd.OutputWindow = hWnd; sd.SampleDesc.Count = 1; sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    D3D_FEATURE_LEVEL fl;
    const D3D_FEATURE_LEVEL lvls[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0 };
    if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, lvls, 2, D3D11_SDK_VERSION,
                                      &sd, &g_pSwapChain, &g_pd3dDevice, &fl, &g_pd3dDeviceContext) != S_OK) return false;
    CreateRenderTarget();
    return true;
}
static void CleanupDeviceD3D() {
    CleanupRenderTarget();
    if (g_pSwapChain) { g_pSwapChain->Release(); g_pSwapChain = nullptr; }
    if (g_pd3dDeviceContext) { g_pd3dDeviceContext->Release(); g_pd3dDeviceContext = nullptr; }
    if (g_pd3dDevice) { g_pd3dDevice->Release(); g_pd3dDevice = nullptr; }
}

// a picture on the graphics card
struct Tex {
    ID3D11ShaderResourceView * srv = nullptr;
    int w = 0, h = 0;
    void drop() { if (srv) { srv->Release(); srv = nullptr; } w = h = 0; }
    ImTextureRef ref() const { return ImTextureRef((ImTextureID) (uintptr_t) srv); }
};
static Tex make_tex(const uint8_t * rgba, int w, int h) {
    Tex t;
    D3D11_TEXTURE2D_DESC d{};
    d.Width = (UINT) w; d.Height = (UINT) h; d.MipLevels = 1; d.ArraySize = 1; d.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    d.SampleDesc.Count = 1; d.Usage = D3D11_USAGE_DEFAULT; d.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd{ rgba, (UINT) w * 4, 0 };
    ID3D11Texture2D * tx = nullptr;
    if (FAILED(g_pd3dDevice->CreateTexture2D(&d, &sd, &tx)) || !tx) return t;
    if (SUCCEEDED(g_pd3dDevice->CreateShaderResourceView(tx, nullptr, &t.srv))) { t.w = w; t.h = h; }
    tx->Release();
    return t;
}

// ── pictures are read off the window's thread (a 2048 x 2048 PNG takes a while; WIC wants its own COM setup) ──
struct LoadReq { std::string path; int max_side = 0; int kind = 0; };          // kind: 0 = the big view, 1 = strip, 2 = reference
struct LoadRes { LoadReq req; int w = 0, h = 0, src_w = 0, src_h = 0; std::vector<uint8_t> rgba; bool ok = false; };      // src: the file's own size
struct Loader {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::deque<LoadReq> in;
    std::deque<LoadRes> out;
    bool stop = false;
    // the picture shrunk (area average) so its longer side is at most max_side, as RGBA
    static void fit(const fluximg::Rgb & im, int max_side, LoadRes & r) {
        const double k = (double) max_side / (double) (std::max)(im.w, im.h);
        r.w = k >= 1.0 ? im.w : (std::max)(1, (int) (im.w * k + 0.5));
        r.h = k >= 1.0 ? im.h : (std::max)(1, (int) (im.h * k + 0.5));
        r.rgba.resize((size_t) r.w * r.h * 4);
        for (int y = 0; y < r.h; y++) {
            const int y0 = (int) ((int64_t) y * im.h / r.h), y1 = (std::max)(y0 + 1, (int) ((int64_t) (y + 1) * im.h / r.h));
            for (int x = 0; x < r.w; x++) {
                const int x0 = (int) ((int64_t) x * im.w / r.w), x1 = (std::max)(x0 + 1, (int) ((int64_t) (x + 1) * im.w / r.w));
                unsigned a[3] = { 0, 0, 0 };
                for (int yy = y0; yy < y1; yy++) {
                    const uint8_t * s = &im.px[((size_t) yy * im.w + x0) * 3];
                    for (int xx = x0; xx < x1; xx++, s += 3) { a[0] += s[0]; a[1] += s[1]; a[2] += s[2]; }
                }
                const unsigned n = (unsigned) ((y1 - y0) * (x1 - x0));
                uint8_t * d = &r.rgba[((size_t) y * r.w + x) * 4];
                d[0] = (uint8_t) ((a[0] + n / 2) / n); d[1] = (uint8_t) ((a[1] + n / 2) / n); d[2] = (uint8_t) ((a[2] + n / 2) / n); d[3] = 255;
            }
        }
    }
    void start() {
        th = std::thread([this]() {
            for (;;) {
                LoadReq q;
                {
                    std::unique_lock<std::mutex> l(m);
                    cv.wait(l, [this]() { return stop || !in.empty(); });
                    if (stop) return;
                    q = in.front(); in.pop_front();
                }
                LoadRes r;
                r.req = q;
                try { const fluximg::Rgb im = fluximg::load(q.path); r.src_w = im.w; r.src_h = im.h; fit(im, q.max_side, r); r.ok = true; } catch (...) { r.ok = false; }
                std::lock_guard<std::mutex> l(m);
                out.push_back(std::move(r));
            }
        });
    }
    void push(const LoadReq & q, bool first = false) {
        { std::lock_guard<std::mutex> l(m); if (first) in.push_front(q); else in.push_back(q); }
        cv.notify_one();
    }
    bool pop(LoadRes & r) {
        std::lock_guard<std::mutex> l(m);
        if (out.empty()) return false;
        r = std::move(out.front()); out.pop_front();
        return true;
    }
    void shutdown() {
        { std::lock_guard<std::mutex> l(m); stop = true; }
        cv.notify_one();
        if (th.joinable()) th.join();
    }
};

// ── window: no native frame. The title bar is ours; Windows still moves, snaps and maximises it (HTCAPTION). ──
static float g_scale    = 1.0f;          // display scale (DPI / 96)
static UINT  g_new_dpi  = 0;             // set by WM_DPICHANGED, applied by the main loop
static int   g_cap_h    = 0;             // title-bar height in client pixels
static RECT  g_nodrag[2] = {};           // title-bar areas that hold controls (language, window buttons)
static std::vector<std::string> g_dropped;      // picture files dropped on the window
extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
static LRESULT WINAPI WndProc(HWND hWnd, UINT msg, WPARAM wP, LPARAM lP) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wP, lP)) return true;
    switch (msg) {
        case WM_SIZE:
            if (g_pd3dDevice && wP != SIZE_MINIMIZED) {
                CleanupRenderTarget();
                g_pSwapChain->ResizeBuffers(0, (UINT) LOWORD(lP), (UINT) HIWORD(lP), DXGI_FORMAT_UNKNOWN, 0);
                CreateRenderTarget();
            }
            return 0;
        case WM_NCCALCSIZE: if (wP) return 0; break;        // remove the native frame
        case WM_NCHITTEST: {
            const LONG b = (LONG) (8 * g_scale);
            POINT pt = { GET_X_LPARAM(lP), GET_Y_LPARAM(lP) };
            RECT r;
            GetWindowRect(hWnd, &r);
            if (!IsZoomed(hWnd)) {
                const bool L = pt.x < r.left + b, R = pt.x >= r.right - b, T = pt.y < r.top + b, B = pt.y >= r.bottom - b;
                if (T && L) return HTTOPLEFT; if (T && R) return HTTOPRIGHT; if (B && L) return HTBOTTOMLEFT; if (B && R) return HTBOTTOMRIGHT;
                if (L) return HTLEFT; if (R) return HTRIGHT; if (T) return HTTOP; if (B) return HTBOTTOM;
            }
            POINT c = pt;
            ScreenToClient(hWnd, &c);
            if (c.y < g_cap_h && !PtInRect(&g_nodrag[0], c) && !PtInRect(&g_nodrag[1], c)) return HTCAPTION;
            return HTCLIENT;
        }
        case WM_GETMINMAXINFO: {                            // maximise inside the work area; a sensible minimum
            MINMAXINFO * mmi = (MINMAXINFO *) lP;
            HMONITOR mon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
            MONITORINFO mi{ sizeof(mi) };
            if (GetMonitorInfo(mon, &mi)) {
                mmi->ptMaxPosition.x = mi.rcWork.left - mi.rcMonitor.left; mmi->ptMaxPosition.y = mi.rcWork.top - mi.rcMonitor.top;
                mmi->ptMaxSize.x = mi.rcWork.right - mi.rcWork.left; mmi->ptMaxSize.y = mi.rcWork.bottom - mi.rcWork.top;
            }
            mmi->ptMinTrackSize.x = (LONG) (900 * g_scale); mmi->ptMinTrackSize.y = (LONG) (620 * g_scale);
            return 0;
        }
        case WM_DPICHANGED: {
            g_new_dpi = HIWORD(wP);
            const RECT * r = (const RECT *) lP;
            SetWindowPos(hWnd, nullptr, r->left, r->top, r->right - r->left, r->bottom - r->top, SWP_NOZORDER | SWP_NOACTIVATE);
            return 0;
        }
        case WM_DROPFILES: {
            HDROP d = (HDROP) wP;
            const UINT n = DragQueryFileW(d, 0xFFFFFFFF, nullptr, 0);
            for (UINT i = 0; i < n; i++) {
                wchar_t b[1024];
                if (DragQueryFileW(d, i, b, 1024)) g_dropped.push_back(narrow(b));
            }
            DragFinish(d);
            return 0;
        }
        case WM_SYSCOMMAND: if ((wP & 0xfff0) == SC_KEYMENU) return 0; break;
        case WM_DESTROY: PostQuitMessage(0); return 0;
    }
    return DefWindowProcW(hWnd, msg, wP, lP);
}

// ════════════════════════ look (PulseX CV's) ════════════════════════
static const float FS = 21.0f;                              // body text size before display scaling
static ImFont * g_font = nullptr, * g_semi = nullptr;
static float px(float v) { return v * g_scale; }
static ImVec4 rgb(int r, int g, int b, float a = 1.0f) { return ImVec4(r / 255.0f, g / 255.0f, b / 255.0f, a); }
static const ImVec4 C_BG    = rgb( 58, 63, 71);            // window: soft slate, not black
static const ImVec4 C_CARD  = rgb( 71, 77, 87);            // cards lift off the window
static const ImVec4 C_INPUT = rgb( 47, 51, 59);            // fields sink into the card
static const ImVec4 C_BTN   = rgb( 92, 99,112);
static const ImVec4 C_BTN_H = rgb(106,114,128);
static const ImVec4 C_TEXT  = rgb(236,239,243);
static const ImVec4 C_DIM   = rgb(170,178,190);
static const ImVec4 C_ACC   = rgb( 42,166,154);            // teal
static const ImVec4 C_ACC_H = rgb( 62,190,177);
static const ImVec4 C_ACC_L = rgb( 34,134,125);
static const ImVec4 C_WARN  = rgb(232,184, 92);
static const ImVec4 C_ERR   = rgb(236,120,110);

static void apply_style() {
    ImGuiStyle st;
    ImGui::StyleColorsDark(&st);
    st.WindowRounding = 0; st.ChildRounding = 20; st.FrameRounding = 12; st.PopupRounding = 12; st.ScrollbarRounding = 10; st.GrabRounding = 10;
    st.WindowPadding = ImVec2(14, 12); st.FramePadding = ImVec2(11, 8); st.ItemSpacing = ImVec2(10, 9); st.ItemInnerSpacing = ImVec2(8, 6);
    st.ScrollbarSize = 11; st.WindowBorderSize = 0; st.ChildBorderSize = 0; st.PopupBorderSize = 1; st.FrameBorderSize = 1;
    ImVec4 * c = st.Colors;
    c[ImGuiCol_WindowBg] = C_BG; c[ImGuiCol_ChildBg] = ImVec4(0, 0, 0, 0); c[ImGuiCol_PopupBg] = rgb(50, 55, 63);
    c[ImGuiCol_Text] = C_TEXT; c[ImGuiCol_TextDisabled] = C_DIM; c[ImGuiCol_Border] = ImVec4(1, 1, 1, 0.07f);
    c[ImGuiCol_FrameBg] = C_INPUT; c[ImGuiCol_FrameBgHovered] = rgb(52, 57, 66); c[ImGuiCol_FrameBgActive] = rgb(52, 57, 66);
    c[ImGuiCol_Button] = C_BTN; c[ImGuiCol_ButtonHovered] = C_BTN_H; c[ImGuiCol_ButtonActive] = rgb(82, 89, 101);
    c[ImGuiCol_Header] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.30f); c[ImGuiCol_HeaderHovered] = ImVec4(1, 1, 1, 0.07f); c[ImGuiCol_HeaderActive] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.45f);
    c[ImGuiCol_CheckMark] = C_ACC_H; c[ImGuiCol_TextSelectedBg] = ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.45f); c[ImGuiCol_InputTextCursor] = C_TEXT;
    c[ImGuiCol_SliderGrab] = C_ACC; c[ImGuiCol_SliderGrabActive] = C_ACC_H;
    c[ImGuiCol_ScrollbarBg] = ImVec4(0, 0, 0, 0); c[ImGuiCol_ScrollbarGrab] = ImVec4(1, 1, 1, 0.16f);
    c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(1, 1, 1, 0.26f); c[ImGuiCol_ScrollbarGrabActive] = ImVec4(1, 1, 1, 0.34f);
    c[ImGuiCol_Separator] = ImVec4(1, 1, 1, 0.08f); c[ImGuiCol_NavCursor] = C_ACC_H;
    c[ImGuiCol_ModalWindowDimBg] = ImVec4(0, 0, 0, 0.45f);
    st.ScaleAllSizes(g_scale); st.FontSizeBase = FS; st.FontScaleDpi = g_scale;
    ImGui::GetStyle() = st;
}

// A rounded panel. size.y == 0 with ImGuiChildFlags_AutoResizeY grows with its content.
static bool begin_card(const char * id, ImVec2 size, const char * title = nullptr, ImGuiChildFlags cf = 0, ImGuiWindowFlags wf = 0) {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, C_CARD);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(20), px(16)));
    const bool v = ImGui::BeginChild(id, size, ImGuiChildFlags_AlwaysUseWindowPadding | cf, wf);
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (title) { ImGui::PushFont(g_semi, FS * 1.08f); ImGui::TextUnformatted(title); ImGui::PopFont(); }
    return v;
}
static void end_card() { ImGui::EndChild(); }
// Small dim caption above a field.
static void label(const char * t) {
    ImGui::PushFont(nullptr, FS * 0.84f);
    ImGui::PushStyleColor(ImGuiCol_Text, C_DIM);
    ImGui::TextUnformatted(t);
    ImGui::PopStyleColor();
    ImGui::PopFont();
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - px(5));
}
static void hint(const char * t, const ImVec4 & col = C_DIM) {
    ImGui::PushFont(nullptr, FS * 0.84f);
    ImGui::PushStyleColor(ImGuiCol_Text, col);
    ImGui::TextWrapped("%s", t);
    ImGui::PopStyleColor();
    ImGui::PopFont();
}
static float lines_h(int n) { return n * ImGui::GetTextLineHeight() + 2 * ImGui::GetStyle().FramePadding.y + px(2); }
static bool primary_button(const char * t, ImVec2 size = ImVec2(0, 0)) {
    ImGui::PushStyleColor(ImGuiCol_Button, C_ACC); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C_ACC_H); ImGui::PushStyleColor(ImGuiCol_ButtonActive, C_ACC_L);
    ImGui::PushFont(g_semi, 0.0f);
    const bool r = ImGui::Button(t, size);
    ImGui::PopFont();
    ImGui::PopStyleColor(3);
    return r;
}
// A pill that is either chosen or not. Chips flow: the next one goes on the same line while it fits.
static bool chip(const char * t, bool on, bool enabled = true) {
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ImGui::GetFrameHeight() * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(px(15), ImGui::GetStyle().FramePadding.y));
    ImGui::PushStyleColor(ImGuiCol_Button, on ? C_ACC : C_INPUT);
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, on ? C_ACC_H : rgb(60, 65, 75));
    ImGui::PushStyleColor(ImGuiCol_ButtonActive, C_ACC_L);
    ImGui::BeginDisabled(!enabled);
    const bool r = ImGui::Button(t);
    ImGui::EndDisabled();
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    return r && enabled;
}
static void chip_flow(const char * next) {
    const float w = ImGui::CalcTextSize(next, nullptr, true).x + 2 * px(15);
    const float right = ImGui::GetCursorScreenPos().x + ImGui::GetContentRegionAvail().x;      // the cursor is at the start of the next line
    if (ImGui::GetItemRectMax().x + ImGui::GetStyle().ItemSpacing.x + w <= right) ImGui::SameLine();
}

// The whole machine draws about 21 W while it makes pictures (measured on battery 10-02 with tools/energy_1002.py:
// 22.0 W Q8 512, 21.4 W Q4 512, 20.2 W Q8 1024; idle 7 W). Energy of a picture = that x its seconds, in Wh.
static const double PICTURE_WATTS = 21.0;
static double watt_hours(double seconds) { return PICTURE_WATTS * seconds / 3600.0; }

// ════════════════════════ the application ════════════════════════
struct SizeOpt { int w = 0, h = 0; bool encoder = false; };       // a compiled decoder; encoder = pictures of this size can be edited
struct Meta { std::string prompt, model, title, sub, enrich, sent; bool top = false; unsigned seed = 0; int w = 0, h = 0, refs = 0, steps = 4, up = 0, bw = 0, bh = 0; double secs = 0; };   // w, h: the file's; up != 0: made bw x bh, then enlarged to the short side `up`
struct Pic { std::string path; uint64_t when = 0; Tex thumb; int state = 0; };      // state: 0 not asked for, 1 asked, 2 there, 3 unreadable
struct RefPic { std::string path; Tex thumb; int state = 0; };
struct Job {
    std::string prompt, model, out;
    int w = 512, h = 512, steps = 4, up = 0;
    unsigned seed = 0;
    std::vector<std::string> refs;
    double est = 23;
    std::string title, sub;                 // text on the finished picture (a poster)
    std::string enrich = "off";             // flux-make --enrich: nouns = Swedish words get their English noun in front
    bool top = false;
    std::string text_src;                   // not empty: no picture is made - the text goes on this existing one
    std::string enlarge_src;                // not empty: no picture is made - this existing one is enlarged to the short side `up`
};
// one argument of a Windows command line, so that the program's argv gets exactly `s` (quotes and trailing backslashes too)
static std::string quote_arg(const std::string & s) {
    std::string o = "\"";
    size_t bs = 0;
    for (char c : s) {
        if (c == '\\') { bs++; continue; }
        if (c == '"') { o.append(bs * 2 + 1, '\\'); o += '"'; bs = 0; continue; }
        o.append(bs, '\\'); bs = 0;
        o += c;
    }
    o.append(bs * 2, '\\');
    return o + "\"";
}

struct App {
    // where things are
    std::string bin, cfg, data_dir, out_dir, vae_qnn;
    std::vector<SizeOpt> sizes;
    bool has_q4 = false, has_q8 = false, has_up = false, ready = false;
    std::string not_ready;                                // why nothing can be made (shown instead of the button)
    // what the user chose
    char prompt[4000] = "";
    char title[160] = "", sub[200] = "";                  // the poster's text; not kept between sessions
    bool text_top = false;
    int size_ix = 0, up = 0, steps = 4, seed = 0;         // up: enlarge the finished picture 2 or 4 times (0 = no)
    std::string model = "q8";
    bool random_seed = true, more = false;
    bool png = false;                                     // save as PNG (exact, three to four times the size) instead of JPEG
    bool swedish = true;                                  // the Norrland theme: help with Swedish words (flux-make --enrich nouns)
    std::string sent;                                     // what flux-make says it really sent, when that is not the prompt as typed
    std::vector<RefPic> refs;
    std::map<std::string, double> measured;               // "q8 512x512 r0 u0" -> seconds the last such picture took
    // the picture being made
    Job job;
    std::thread th;
    std::mutex m;
    std::string stage, log;
    int step = 0, nsteps = 4;
    double t0 = 0, t_stage = 0;
    std::atomic<bool> running{ false }, finished{ false };
    int rc = 0;
    std::string error, error_detail;
    std::string busy_name;                                // another program on the NPU: ask before starting
    bool busy_enlarge = false;                            // ... and what was being started was "Enlarge", not a new picture
    // pictures
    Loader loader;
    std::vector<Pic> pics;
    std::map<std::string, Meta> meta;
    std::string sel, view_path;
    Tex view;
    int view_w = 0, view_h = 0;                           // the shown picture's own size (the texture may be a reduced copy)
    bool view_asked = false;

    std::string settings_file() const { return data_dir + "\\settings.txt"; }
    std::string history_file() const { return data_dir + "\\history.tsv"; }
    const SizeOpt & size() const { return sizes[(size_t) size_ix]; }

    void find_things() {
        bin = app_dir();
        const char * e = getenv("FLUX_CONFIG");
        cfg = e && *e ? e : bin + "\\flux.json";
        const char * ed = getenv("PULSEX_IMAGE_DATA"), * eo = getenv("PULSEX_IMAGE_OUT");      // another settings / pictures folder (checks)
        if (ed && *ed) data_dir = ed;
        else {
            data_dir = known_folder(FOLDERID_LocalAppData);
            if (data_dir.empty()) data_dir = bin;
            data_dir += "\\PulseX Image";
        }
        CreateDirectoryW(to_w(data_dir).c_str(), nullptr);
        if (eo && *eo) out_dir = eo;
        else {
            out_dir = known_folder(FOLDERID_Pictures);
            if (out_dir.empty()) out_dir = data_dir;
            out_dir += "\\PulseX";
        }
        CreateDirectoryW(to_w(out_dir).c_str(), nullptr);

        const std::string js = read_all(cfg);
        // a path in flux.json: absolute as it is, relative from the folder flux.json is in (as flux-make reads it)
        auto full = [](const std::string & p) { wchar_t b[2048]; const DWORD n = GetFullPathNameW(to_w(p).c_str(), 2048, b, nullptr); return n && n < 2048 ? narrow(b) : p; };
        const std::string cfg_full = full(cfg), base = cfg_full.substr(0, cfg_full.find_last_of("\\/"));
        auto at = [&](const std::string & key) {
            const std::string p = json_str(js, key);
            if (p.empty()) return p;
            return full(((p.size() > 1 && p[1] == ':') || p[0] == '\\') ? p : base + "\\" + p);
        };
        vae_qnn = at("vae_qnn");
        has_q4 = file_exists(at("transformer_q4")) && file_exists(at("encoder_q4"));
        has_q8 = file_exists(at("transformer_q8")) && file_exists(at("encoder_q8"));
        std::string sr = at("quicksrnet_x4");
        if (sr.size() > 5 && sr.substr(sr.size() - 5) == ".onnx") sr = sr.substr(0, sr.size() - 5) + "_ctx_qnn.bin";
        has_up = file_exists(sr);
        sizes.clear();
        WIN32_FIND_DATAW fd;
        HANDLE h = FindFirstFileW(to_w(vae_qnn + "\\flux2-vae-decoder-*_ctx_qnn.bin").c_str(), &fd);
        if (h != INVALID_HANDLE_VALUE) {
            do {
                const std::string n = narrow(fd.cFileName);
                SizeOpt s;
                if (sscanf(n.c_str(), "flux2-vae-decoder-%dx%d_ctx_qnn.bin", &s.w, &s.h) == 2 && s.w >= 64 && s.h >= 64) {
                    s.encoder = file_exists(vae_qnn + "\\flux2-vae-encoder-" + std::to_string(s.w) + "x" + std::to_string(s.h) + "_ctx_qnn.bin");
                    sizes.push_back(s);
                }
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        std::sort(sizes.begin(), sizes.end(), [](const SizeOpt & a, const SizeOpt & b) { return a.w * a.h != b.w * b.h ? a.w * a.h < b.w * b.h : a.w < b.w; });
        if (!file_exists(bin + "\\flux-make.exe")) not_ready = tr("Programmet flux-make.exe saknas bredvid appen.", "flux-make.exe is missing next to the app.");
        else if (js.empty()) not_ready = tr("Inställningsfilen flux.json hittades inte.", "The settings file flux.json was not found.");
        else if (!has_q4 && !has_q8) not_ready = tr("Modellfilerna hittades inte (se flux.json).", "The model files were not found (see flux.json).");
        else if (sizes.empty() || !file_exists(vae_qnn + "\\bn.f32")) not_ready = tr("Bildavkodaren saknas (mappen vae-qnn).", "The picture decoder is missing (the vae-qnn folder).");
        ready = not_ready.empty();
        if (sizes.empty()) sizes.push_back(SizeOpt{ 512, 512, false });
        if (model == "q8" && !has_q8) model = "q4";
        if (model == "q4" && !has_q4) model = "q8";
    }

    void settings_load() {
        for (const std::string & line : split(read_all(settings_file()), '\n')) {
            const size_t eq = line.find('=');
            if (eq == std::string::npos) continue;
            const std::string k = line.substr(0, eq), v = unesc(line.substr(eq + 1));
            if (k == "lang") g_lang = atoi(v.c_str()) ? 1 : 0;
            else if (k == "model") model = v == "q4" ? "q4" : "q8";
            else if (k == "size") { for (size_t i = 0; i < sizes.size(); i++) if (std::to_string(sizes[i].w) + "x" + std::to_string(sizes[i].h) == v) size_ix = (int) i; }
            else if (k == "upx") up = atoi(v.c_str()) == 4 ? 4 : atoi(v.c_str()) == 2 ? 2 : 0;
            else if (k == "steps") steps = (std::min)(8, (std::max)(1, atoi(v.c_str())));
            else if (k == "seed") seed = atoi(v.c_str());
            else if (k == "random") random_seed = atoi(v.c_str()) != 0;
            else if (k == "swedish") swedish = atoi(v.c_str()) != 0;
            else if (k == "png") png = atoi(v.c_str()) != 0;
            else if (k == "prompt") snprintf(prompt, sizeof prompt, "%s", v.c_str());
            else if (k.compare(0, 5, "time ") == 0) measured[k.substr(5)] = atof(v.c_str());
        }
        if (model == "q8" && !has_q8) model = "q4";
        if (model == "q4" && !has_q4) model = "q8";
    }
    bool test_run = false;                                // --shot / --demo-run: the user's settings are left as they were
    void settings_save() {
        if (test_run) return;
        std::string s ="lang=" + std::to_string(g_lang) + "\nmodel=" + model + "\nsize=" + std::to_string(size().w) + "x" + std::to_string(size().h) +
                        "\nupx=" + std::to_string(up) + "\nsteps=" + std::to_string(steps) + "\nseed=" + std::to_string(seed) + "\nrandom=" + std::to_string(random_seed ? 1 : 0) + "\nswedish=" + std::to_string(swedish ? 1 : 0) + "\npng=" + std::to_string(png ? 1 : 0) +
                        "\nprompt=" + esc(prompt) + "\n";
        for (const auto & kv : measured) { char b[32]; snprintf(b, sizeof b, "%.1f", kv.second); s += "time " + kv.first + "=" + b + "\n"; }
        write_all(settings_file(), s);
    }
    void history_load() {
        for (const std::string & line : split(read_all(history_file()), '\n')) {
            const std::vector<std::string> f = split(line, '\t');
            if (f.size() < 9) continue;
            Meta mt;
            mt.seed = (unsigned) strtoul(f[1].c_str(), nullptr, 10); mt.w = atoi(f[2].c_str()); mt.h = atoi(f[3].c_str()); mt.model = f[4];
            mt.secs = atof(f[5].c_str()); mt.refs = atoi(f[6].c_str()); mt.steps = atoi(f[7].c_str()); mt.prompt = unesc(f[8]);
            if (f.size() > 9) mt.up = atoi(f[9].c_str());
            if (f.size() > 11) { mt.title = unesc(f[10]); mt.sub = unesc(f[11]); }
            if (f.size() > 12) mt.top = atoi(f[12].c_str()) != 0;
            if (f.size() > 14) { mt.bw = atoi(f[13].c_str()); mt.bh = atoi(f[14].c_str()); }
            if (f.size() > 16) { mt.enrich = f[15]; mt.sent = unesc(f[16]); }
            meta[name_of(f[0])] = mt;                      // by file name (older lines hold the full path)
        }
    }
    void history_add(const std::string & path, const Meta & mt) {
        meta[name_of(path)] = mt;
        char b[64];
        snprintf(b, sizeof b, "%.1f", mt.secs);
        write_all(history_file(), path + "\t" + std::to_string(mt.seed) + "\t" + std::to_string(mt.w) + "\t" + std::to_string(mt.h) + "\t" + mt.model + "\t" + b + "\t" +
                                  std::to_string(mt.refs) + "\t" + std::to_string(mt.steps) + "\t" + esc(mt.prompt) + "\t" + std::to_string(mt.up) + "\t" + esc(mt.title) + "\t" + esc(mt.sub) + "\t" + (mt.top ? "1" : "0") + "\t" + std::to_string(mt.bw) + "\t" + std::to_string(mt.bh) + "\t" + mt.enrich + "\t" + esc(mt.sent) + "\n", L"ab");
    }

    // the pictures in the output folder, newest first (the strip under the big picture)
    void scan_pics() {
        std::vector<Pic> found;
        WIN32_FIND_DATAW fd;
        for (const char * pattern : { "\\*.jpg", "\\*.png" }) {
            HANDLE h = FindFirstFileW(to_w(out_dir + pattern).c_str(), &fd);
            if (h == INVALID_HANDLE_VALUE) continue;
            do {
                Pic p;
                p.path = out_dir + "\\" + narrow(fd.cFileName);
                p.when = ((uint64_t) fd.ftLastWriteTime.dwHighDateTime << 32) | fd.ftLastWriteTime.dwLowDateTime;
                found.push_back(p);
            } while (FindNextFileW(h, &fd));
            FindClose(h);
        }
        std::sort(found.begin(), found.end(), [](const Pic & a, const Pic & b) { return a.when > b.when; });
        if (found.size() > 60) found.resize(60);
        for (Pic & p : found)
            for (Pic & old : pics)
                if (old.path == p.path && old.when == p.when) { p.thumb = old.thumb; p.state = old.state; old.thumb = Tex(); }
        for (Pic & old : pics) old.thumb.drop();
        pics.swap(found);
    }
    void select(const std::string & path) {
        if (sel == path) return;
        sel = path;
        view_asked = false;
    }

    // ── the estimate: measured on this machine (512 x 512 Q4 20 s, 768 x 512 27 s, 1024 x 1024 63 s; Q8 about 13 % more;
    //    a reference picture adds its own tokens), replaced by what the last picture of the same kind really took ──
    std::string est_key(const std::string & mdl, int w, int h, int nref, int upv, int st) const {
        return mdl + " " + std::to_string(w) + "x" + std::to_string(h) + " r" + std::to_string(nref) + " u" + std::to_string(upv ? 1 : 0) + " s" + std::to_string(st);
    }
    double estimate(const std::string & mdl, int w, int h, int nref, int upv, int st) const {
        const auto it = measured.find(est_key(mdl, w, h, nref, upv, st));
        if (it != measured.end()) return it->second;
        const double tok = (double) (w / 16) * (h / 16) * (1 + 0.75 * nref);       // 2 references, Q8, 512 x 512: 44 s measured
        return 5.7 + 0.014 * tok * (mdl == "q8" ? 1.13 : 1.0) * st / 4.0 + (nref ? 1.5 : 0.0) + (upv ? 0.6 : 0.0);
    }
    // the enlarged picture's short side for the chosen shape and factor (what flux-make --up takes), 0 = not enlarged
    int up_for(int factor) const { const int m = (std::min)(size().w, size().h); return has_up && factor && m >= 512 && m * factor <= 2048 ? m * factor : 0; }
    int up_now() const { return up_for(up); }

    // another picture / video / language engine on the NPU right now? Its name, or "".
    static std::string npu_busy() {
        std::string hit;
        HANDLE s = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
        if (s == INVALID_HANDLE_VALUE) return hit;
        PROCESSENTRY32W pe{ sizeof pe };
        if (Process32FirstW(s, &pe))
            do {
                std::string n = narrow(pe.szExeFile);
                for (char & c : n) c = (char) tolower((unsigned char) c);
                for (const char * k : { "flux-make", "flux-stream", "flux-encode", "flux-vae", "flux-qnn", "zimage", "ltx", "llama-server", "test-backend-ops" })
                    if (n.compare(0, strlen(k), k) == 0) hit = n;
            } while (hit.empty() && Process32NextW(s, &pe));
        CloseHandle(s);
        return hit;
    }

    std::string new_out_path() const {
        SYSTEMTIME st;
        GetLocalTime(&st);
        char b[80];
        snprintf(b, sizeof b, "pulsex_%04d%02d%02d_%02d%02d%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
        const std::string ext = png ? ".png" : ".jpg";        // .jpg: flux-make writes a JPEG, quality 95
        std::string p = out_dir + "\\" + b + ext;
        for (int i = 2; file_exists(p); i++) p = out_dir + "\\" + b + "_" + std::to_string(i) + ext;      // never over a picture that is there
        return p;
    }

    void start(bool even_if_busy = false) {
        if (running || !ready || !prompt[0]) return;
        if (!even_if_busy) { busy_name = npu_busy(); if (!busy_name.empty()) { busy_enlarge = false; return; } }
        if (random_seed) { std::random_device rd; seed = (int) (rd() % 1000000u); }
        error.clear(); error_detail.clear();
        job = Job();
        job.prompt = prompt; job.model = model; job.w = size().w; job.h = size().h; job.steps = steps; job.up = up_now(); job.seed = (unsigned) seed;
        for (const RefPic & r : refs) job.refs.push_back(r.path);
        job.title = title; job.sub = sub; job.top = text_top;
        job.enrich = swedish ? "nouns" : "off";
        sent.clear();
        job.out = new_out_path();
        job.est = estimate(job.model, job.w, job.h, (int) job.refs.size(), job.up, job.steps);
        { std::lock_guard<std::mutex> l(m); stage = "text"; step = 0; nsteps = job.steps; log.clear(); t0 = t_stage = now_s(); }
        finished = false; running = true;
        if (th.joinable()) th.join();
        th = std::thread([this]() { work(); });
    }
    // the text on the picture that is shown: a new file beside it, the NPU is not used
    void start_text() {
        if (running || sel.empty() || (!title[0] && !sub[0])) return;
        error.clear(); error_detail.clear();
        job = Job();
        job.text_src = sel; job.title = title; job.sub = sub; job.top = text_top;
        job.out = new_out_path();
        job.est = 1;
        { std::lock_guard<std::mutex> l(m); stage = "addtext"; step = 0; nsteps = 1; log.clear(); t0 = t_stage = now_s(); }
        finished = false; running = true;
        if (th.joinable()) th.join();
        th = std::thread([this]() { work(); });
    }

    // the shown picture's short side after enlarging it twice, 0 when it cannot be (too small, too large, no upscaler, not loaded yet)
    int enlarge_to() const {
        if (!has_up || sel.empty() || view_path != sel || !view_w) return 0;
        const int m = (std::min)(view_w, view_h);
        return m >= 512 && m * 2 <= 2048 && (std::max)(view_w, view_h) <= 1536 ? m * 2 : 0;
    }
    // the picture that is shown, enlarged: a new file, the upscaler alone on the NPU (about a second)
    void start_enlarge(bool even_if_busy = false) {
        const int to = enlarge_to();
        if (running || !to) return;
        if (!even_if_busy) { busy_name = npu_busy(); if (!busy_name.empty()) { busy_enlarge = true; return; } }
        error.clear(); error_detail.clear();
        job = Job();
        job.enlarge_src = sel; job.up = to; job.w = view_w; job.h = view_h;
        job.out = new_out_path();
        job.est = 2;
        { std::lock_guard<std::mutex> l(m); stage = "upscale"; step = 0; nsteps = 1; log.clear(); t0 = t_stage = now_s(); }
        finished = false; running = true;
        if (th.joinable()) th.join();
        th = std::thread([this]() { work(); });
    }

    // the worker: flux-make.exe, its stdout and stderr through one pipe, line by line
    void work() {
        const auto & q = quote_arg;
        std::string cmd = q(bin + "\\flux-make.exe");
        if (!job.text_src.empty()) cmd += " --add-text " + q(job.text_src) + " --out " + q(job.out);
        else if (!job.enlarge_src.empty()) cmd += " --enlarge " + q(job.enlarge_src) + " --up " + std::to_string(job.up) + " --progress --config " + q(cfg) + " --out " + q(job.out);
        else {
            const std::string pf = data_dir + "\\prompt.txt";
            write_all(pf, job.prompt);
            cmd += " --prompt-file " + q(pf) + " --progress --config " + q(cfg) + " --size " + std::to_string(job.w) + "x" + std::to_string(job.h) +
                   " --seed " + std::to_string(job.seed) + " --steps " + std::to_string(job.steps) + " --model " + job.model + " --out " + q(job.out);
            if (job.up) cmd += " --up " + std::to_string(job.up);
            if (job.enrich != "off") cmd += " --enrich " + job.enrich;
            for (const std::string & r : job.refs) cmd += " --image " + q(r);
        }
        if (!job.title.empty()) cmd += " --title " + q(job.title);
        if (!job.sub.empty()) cmd += " --sub " + q(job.sub);
        if (job.top && (!job.title.empty() || !job.sub.empty())) cmd += " --top";
        std::wstring w = to_w(cmd);
        SECURITY_ATTRIBUTES sa{ sizeof sa, nullptr, TRUE };
        HANDLE rd = nullptr, wr = nullptr;
        int code = 100;
        if (CreatePipe(&rd, &wr, &sa, 0)) {
            SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
            STARTUPINFOW si{};
            si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES; si.hStdOutput = wr; si.hStdError = wr;
            PROCESS_INFORMATION pi{};
            const BOOL ok = CreateProcessW(nullptr, &w[0], nullptr, nullptr, TRUE, CREATE_NO_WINDOW, nullptr, to_w(bin).c_str(), &si, &pi);      // cwd = the engines' folder
            CloseHandle(wr);
            if (ok) {
                std::string line;
                char b[1024];
                DWORD n = 0;
                while (ReadFile(rd, b, sizeof b, &n, nullptr) && n)
                    for (DWORD i = 0; i < n; i++) {
                        if (b[i] == '\n') { on_line(line); line.clear(); }
                        else if (b[i] != '\r') line += b[i];
                    }
                if (!line.empty()) on_line(line);
                WaitForSingleObject(pi.hProcess, INFINITE);
                DWORD ec = 1;
                GetExitCodeProcess(pi.hProcess, &ec);
                code = (int) ec;
                CloseHandle(pi.hProcess); CloseHandle(pi.hThread);
            }
            CloseHandle(rd);
        }
        rc = code;
        finished = true;
    }
    void on_line(const std::string & line) {
        std::lock_guard<std::mutex> l(m);
        if (line.compare(0, 6, "sent: ") == 0) { sent = line.substr(6); return; }
        if (line.compare(0, 9, "PROGRESS ") != 0) { log += line + "\n"; return; }
        const std::vector<std::string> f = split(line.substr(9), ' ');
        stage = f[0];
        if (stage == "step" && f.size() >= 3) { step = atoi(f[1].c_str()); nsteps = (std::max)(1, atoi(f[2].c_str())); }
        t_stage = now_s();
    }
    // how far along (0..1) and what to call the stage
    float fraction(std::string & what) {
        std::lock_guard<std::mutex> l(m);
        const double e_text = 3.0, e_ref = job.refs.empty() ? 0.0 : 1.5, e_pic = 1.2 + (job.up ? 0.6 : 0.0);
        const double e_steps = (std::max)(4.0, job.est - e_text - e_ref - e_pic), total = e_text + e_ref + e_steps + e_pic;
        const double el = now_s() - t_stage;
        double before = 0, cur = e_text;
        char b[96];
        if (stage == "addtext") { what = tr("Skriver texten …", "Writing the text …"); return (float) (std::min)(0.95, el / 1.0); }
        if (!job.enlarge_src.empty()) { what = tr("Förstorar …", "Enlarging …"); return (float) (std::min)(0.95, (now_s() - t0) / 2.0); }
        if (stage == "text") what = tr("Läser beskrivningen …", "Reading the description …");
        else if (stage == "reference") { before = e_text; cur = e_ref; what = tr("Tittar på din bild …", "Looking at your picture …"); }
        else if (stage == "step") {
            before = e_text + e_ref + e_steps * step / nsteps; cur = e_steps / nsteps;
            snprintf(b, sizeof b, tr("Målar – steg %d av %d", "Painting – step %d of %d"), (std::min)(step + 1, nsteps), nsteps);
            what = b;
        }
        else if (stage == "upscale") { before = e_text + e_ref + e_steps + 1.2; cur = 0.6; what = tr("Förstorar …", "Enlarging …"); }
        else { before = e_text + e_ref + e_steps; cur = e_pic; what = tr("Framkallar bilden …", "Developing the picture …"); }
        return (float) ((before + (std::min)(el, cur * 0.95)) / total);
    }

    void add_ref(const std::string & path) {
        if (!is_picture_file(path) || refs.size() >= 2) return;
        for (const RefPic & r : refs) if (r.path == path) return;
        RefPic r;
        r.path = path;
        refs.push_back(r);
        if (!size().encoder)                                // editing needs the size's encoder: move to a size that has one
            for (size_t i = 0; i < sizes.size(); i++) if (sizes[i].encoder) { size_ix = (int) i; break; }
    }
    bool can_edit() const { for (const SizeOpt & s : sizes) if (s.encoder) return true; return false; }

    // once per frame: finished pictures, loaded textures, dropped files
    void pump() {
        if (running && finished) {
            if (th.joinable()) th.join();
            running = false;
            const double secs = now_s() - t0;
            if (rc == 0 && file_exists(job.out)) {
                Meta mt;
                if (!job.text_src.empty()) {                // the text on an existing picture: what that picture was made from, plus the text
                    const auto src = meta.find(name_of(job.text_src));
                    if (src != meta.end()) mt = src->second;
                } else if (!job.enlarge_src.empty()) {      // an existing picture enlarged: what it was made from, at its new size
                    const auto src = meta.find(name_of(job.enlarge_src));
                    if (src != meta.end()) mt = src->second;
                    if (!mt.bw) { mt.bw = job.w; mt.bh = job.h; }
                    mt.w = job.w * 2; mt.h = job.h * 2; mt.up = job.up;
                    job.title = mt.title; job.sub = mt.sub; job.top = mt.top;       // nothing was written now: keep what the picture had
                } else {
                    mt.prompt = job.prompt; mt.model = job.model; mt.seed = job.seed; mt.bw = job.w; mt.bh = job.h;
                    { const int m = (std::min)(job.w, job.h); mt.w = job.up ? job.w * job.up / m : job.w; mt.h = job.up ? job.h * job.up / m : job.h; }
                    mt.refs = (int) job.refs.size(); mt.steps = job.steps; mt.secs = secs; mt.up = job.up;
                    mt.enrich = job.enrich; { std::lock_guard<std::mutex> l(m); mt.sent = sent; }
                    measured[est_key(job.model, job.w, job.h, (int) job.refs.size(), job.up, job.steps)] = secs;
                }
                mt.title = job.title; mt.sub = job.sub; mt.top = job.top;
                history_add(job.out, mt);
                scan_pics();
                select(job.out);
            } else {
                std::lock_guard<std::mutex> l(m);
                error = tr("Det gick inte att skapa bilden.", "The picture could not be made.");
                error_detail = log.size() > 1500 ? log.substr(log.size() - 1500) : log;
                if (rc == 100 && error_detail.empty()) error_detail = "flux-make.exe could not be started";
            }
            settings_save();
        }
        for (const std::string & f : g_dropped) add_ref(f);
        g_dropped.clear();
        LoadRes r;
        while (loader.pop(r)) {
            Tex t = r.ok ? make_tex(r.rgba.data(), r.w, r.h) : Tex();
            bool used = false;
            if (r.req.kind == 0 && r.req.path == sel) { view.drop(); view = t; view_path = sel; view_w = r.src_w; view_h = r.src_h; used = true; }
            if (r.req.kind == 1) for (Pic & p : pics) if (p.path == r.req.path && p.state == 1) { p.thumb = t; p.state = t.srv ? 2 : 3; used = true; break; }
            if (r.req.kind == 2) for (RefPic & p : refs) if (p.path == r.req.path && p.state == 1) { p.thumb = t; p.state = t.srv ? 2 : 3; used = true; break; }
            if (!used) t.drop();
        }
        if (!sel.empty() && !view_asked) { view_asked = true; loader.push(LoadReq{ sel, 1600, 0 }, true); }
    }
};

// a picture drawn to cover a square (the middle of it), with rounded corners
static void draw_cover(ImDrawList * dl, const Tex & t, ImVec2 p, float side, float round) {
    ImVec2 u0(0, 0), u1(1, 1);
    if (t.w > t.h) { const float c = (1.0f - (float) t.h / t.w) * 0.5f; u0.x = c; u1.x = 1 - c; }
    else if (t.h > t.w) { const float c = (1.0f - (float) t.w / t.h) * 0.5f; u0.y = c; u1.y = 1 - c; }
    dl->AddImageRounded(t.ref(), p, ImVec2(p.x + side, p.y + side), u0, u1, IM_COL32_WHITE, round);
}

static std::string size_name(const SizeOpt & s) {
    if (s.w == s.h) return s.w <= 512 ? tr("Kvadrat", "Square") : tr("Stor kvadrat", "Large square");
    return s.w > s.h ? tr("Liggande", "Landscape") : tr("Stående", "Portrait");
}

static bool pick_picture(HWND owner, std::string & out) {
    wchar_t file[1024] = L"";
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof ofn; ofn.hwndOwner = owner; ofn.lpstrFile = file; ofn.nMaxFile = 1024;
    ofn.lpstrFilter = L"Pictures\0*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff;*.gif\0\0";
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;
    if (!GetOpenFileNameW(&ofn)) return false;
    out = narrow(file);
    return true;
}

// ── left: what to make ──
static void draw_left(App & a, HWND hWnd) {
    const float action_h = px(104);
    const ImGuiStyle & st = ImGui::GetStyle();
    ImGui::BeginChild("##choices", ImVec2(0, ImGui::GetContentRegionAvail().y - action_h - st.ItemSpacing.y), 0, 0);
    ImGui::BeginDisabled(a.running);

    begin_card("##prompt", ImVec2(0, 0), tr("Beskriv bilden", "Describe the picture"), ImGuiChildFlags_AutoResizeY);
    ImGui::InputTextMultiline("##p", a.prompt, sizeof a.prompt, ImVec2(-1, lines_h(5)), ImGuiInputTextFlags_WordWrap);
    if (!a.prompt[0]) {
        hint(tr("Prova till exempel:", "Try for example:"));
        static const char * EX[] = { "A moose in a snowy spruce forest under the northern lights",          // the Norrland motifs of zimage
                                     "Reindeer grazing on the autumn mountain tundra",
                                     "The rapids of a northern river under the midnight sun",
                                     "A small cafe with a paper star in the window, winter evening" };
        ImGui::PushFont(nullptr, FS * 0.90f);
        for (const char * e : EX) if (ImGui::Selectable(e)) snprintf(a.prompt, sizeof a.prompt, "%s", e);
        ImGui::PopFont();
    }
    end_card();

    begin_card("##look", ImVec2(0, 0), tr("Format och kvalitet", "Shape and quality"), ImGuiChildFlags_AutoResizeY);
    label(tr("Format", "Shape"));
    for (size_t i = 0; i < a.sizes.size(); i++) {
        const SizeOpt & s = a.sizes[i];
        const std::string nm = size_name(s) + "###size" + std::to_string(i);
        if (i) chip_flow(nm.c_str());
        const bool usable = a.refs.empty() || s.encoder;
        if (chip(nm.c_str(), (int) i == a.size_ix, usable)) a.size_ix = (int) i;
        if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
            ImGui::SetTooltip("%s", tr("Det här formatet går inte att använda när du utgår från en egen bild.", "This shape cannot be used when you start from a picture of your own."));
    }
    if (a.has_q4 && a.has_q8) {
        label(tr("Kvalitet", "Quality"));
        if (chip(tr("Bäst###q8", "Best###q8"), a.model == "q8")) a.model = "q8";
        chip_flow(tr("Snabbare", "Faster"));
        if (chip(tr("Snabbare###q4", "Faster###q4"), a.model == "q4")) a.model = "q4";
    }
    if (a.up_for(2)) {
        label(tr("Förstora den färdiga bilden", "Enlarge the finished picture"));
        if (chip(tr("Nej###u0", "No###u0"), a.up_now() == 0)) a.up = 0;
        chip_flow("2\xC3\x97");
        if (chip("2\xC3\x97###u2", a.up == 2)) a.up = 2;
        if (a.up_for(4)) {
            chip_flow("4\xC3\x97");
            if (chip("4\xC3\x97###u4", a.up == 4)) a.up = 4;
        }
    }
    {
        const int upv = a.up_now(), m = (std::min)(a.size().w, a.size().h);
        const int w = upv ? a.size().w * upv / m : a.size().w, h = upv ? a.size().h * upv / m : a.size().h;
        char b[200];
        const double est = a.estimate(a.model, a.size().w, a.size().h, (int) a.refs.size(), upv, a.steps);
        snprintf(b, sizeof b, tr("%d × %d punkter · ungefär %.0f sekunder · ungefär %.2f Wh", "%d × %d pixels · about %.0f seconds · about %.2f Wh"), w, h, est, watt_hours(est));
        hint(b);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Energin för hela datorn medan bilden görs (cirka 21 W, mätt på batteri).",
                                                             "The energy of the whole computer while the picture is made (about 21 W, measured on battery)."));
    }
    end_card();

    // a poster: a title and a smaller line, written on the finished picture (spelled as typed - the model is not asked to draw letters)
    begin_card("##text", ImVec2(0, 0), tr("Text på bilden (valfritt)", "Text on the picture (optional)"), ImGuiChildFlags_AutoResizeY);
    label(tr("Rubrik", "Title"));
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##title", a.title, sizeof a.title);
    label(tr("Mindre rad under", "Smaller line under it"));
    ImGui::SetNextItemWidth(-1);
    ImGui::InputText("##sub", a.sub, sizeof a.sub);
    if (a.title[0] || a.sub[0]) {
        if (chip(tr("Nederst###tb", "At the bottom###tb"), !a.text_top)) a.text_top = false;
        chip_flow(tr("Överst", "At the top"));
        if (chip(tr("Överst###tt", "At the top###tt"), a.text_top)) a.text_top = true;
    }
    end_card();

    if (a.can_edit()) {
        begin_card("##refs", ImVec2(0, 0), tr("Utgå från en egen bild (valfritt)", "Start from a picture of your own (optional)"), ImGuiChildFlags_AutoResizeY);
        const float side = px(76);
        ImDrawList * dl = ImGui::GetWindowDrawList();
        int remove = -1;
        for (size_t i = 0; i < a.refs.size(); i++) {
            RefPic & r = a.refs[i];
            if (r.state == 0) { r.state = 1; a.loader.push(LoadReq{ r.path, 256, 2 }); }
            ImGui::PushID((int) i);
            const ImVec2 p = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("##ref", ImVec2(side, side));
            dl->AddRectFilled(p, ImVec2(p.x + side, p.y + side), ImGui::GetColorU32(C_INPUT), px(12));
            if (r.state == 2) draw_cover(dl, r.thumb, p, side, px(12));
            else if (r.state == 3) dl->AddText(ImVec2(p.x + px(10), p.y + side * 0.5f - ImGui::GetTextLineHeight() * 0.5f), ImGui::GetColorU32(C_ERR), "?");
            ImGui::SameLine(0, px(4));
            ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, px(10));
            if (ImGui::Button("\xC3\x97", ImVec2(px(30), px(30)))) remove = (int) i;          // multiplication sign as the remove mark
            ImGui::PopStyleVar();
            ImGui::PopID();
            ImGui::SameLine(0, px(12));
        }
        if (remove >= 0) { a.refs[(size_t) remove].thumb.drop(); a.refs.erase(a.refs.begin() + remove); }
        if (a.refs.size() < 2) {
            if (ImGui::Button(tr("Välj bild …", "Choose a picture …"), ImVec2(0, a.refs.empty() ? 0 : side))) { std::string f; if (pick_picture(hWnd, f)) a.add_ref(f); }
        } else ImGui::NewLine();
        hint(a.refs.empty() ? tr("Du kan också dra en bild hit. Beskriv sedan vad som ska ändras – till exempel \"gör himlen rosa\".",
                                 "You can also drop a picture here. Then describe what to change – for example \"make the sky pink\".")
                            : tr("Beskriv ovan vad som ska ändras i bilden.", "Describe above what to change in the picture."));
        end_card();
    }

    begin_card("##more", ImVec2(0, 0), nullptr, ImGuiChildFlags_AutoResizeY);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, st.FramePadding.y * 0.4f));
    if (ImGui::Button(a.more ? tr("Färre inställningar###more", "Fewer settings###more") : tr("Fler inställningar###more", "More settings###more"))) a.more = !a.more;
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    if (a.more) {
        ImGui::Checkbox(tr("Förstå svenska ord (älg, fors, norrsken …)", "Understand Swedish words (älg, fors, norrsken …)"), &a.swedish);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Lägger det engelska ordet först i beskrivningen. Utan svenska ord skickas den som den är skriven.",
                                                             "Puts the English word first in the description. Without Swedish words it is sent as typed."));
        ImGui::Checkbox(tr("Spara som PNG i stället för JPG", "Save as PNG instead of JPG"), &a.png);
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("PNG sparar bilden exakt men blir tre till fyra gånger så stor. JPG (kvalitet 95) ser likadan ut.",
                                                             "PNG keeps the picture exact but is three to four times the size. JPG (quality 95) looks the same."));
        ImGui::Checkbox(tr("Ny variant varje gång", "A new variation every time"), &a.random_seed);
        label(tr("Variantnummer (samma nummer och beskrivning ger samma bild)", "Variation number (the same number and description give the same picture)"));
        ImGui::BeginDisabled(a.random_seed);
        ImGui::SetNextItemWidth(px(170));
        ImGui::InputInt("##seed", &a.seed, 0, 0);
        if (a.seed < 0) a.seed = 0;
        ImGui::EndDisabled();
        label(tr("Antal steg (4 är modellens eget val)", "Number of steps (4 is the model's own choice)"));
        ImGui::SetNextItemWidth(px(170));
        ImGui::SliderInt("##steps", &a.steps, 2, 8);
    }
    end_card();

    ImGui::EndDisabled();
    ImGui::EndChild();

    // the button, or how far the picture has come
    ImGui::BeginChild("##action", ImVec2(0, action_h), 0, ImGuiWindowFlags_NoScrollbar);
    if (a.running) {
        std::string what;
        const float f = a.fraction(what);
        ImGui::PushFont(g_semi, 0.0f);
        ImGui::TextUnformatted(what.c_str());
        ImGui::PopFont();
        char b[64];
        snprintf(b, sizeof b, tr("%.0f s av ungefär %.0f", "%.0f s of about %.0f"), now_s() - a.t0, a.job.est);
        ImGui::SameLine(ImGui::GetContentRegionAvail().x - ImGui::CalcTextSize(b).x);
        ImGui::TextColored(C_DIM, "%s", b);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        const float w = ImGui::GetContentRegionAvail().x, h = px(12);
        ImDrawList * dl = ImGui::GetWindowDrawList();
        dl->AddRectFilled(p, ImVec2(p.x + w, p.y + h), ImGui::GetColorU32(C_INPUT), h * 0.5f);
        dl->AddRectFilled(p, ImVec2(p.x + (std::max)(h, w * f), p.y + h), ImGui::GetColorU32(C_ACC), h * 0.5f);
        ImGui::Dummy(ImVec2(w, h));
    } else if (!a.ready) {
        hint(a.not_ready.c_str(), C_WARN);
    } else {
        ImGui::BeginDisabled(!a.prompt[0]);
        if (primary_button(a.refs.empty() ? tr("Skapa bild", "Create picture") : tr("Ändra bilden", "Change the picture"), ImVec2(-1, px(54)))) a.start();
        ImGui::EndDisabled();
        if (!a.error.empty()) {
            hint(a.error.c_str(), C_ERR);
            if (ImGui::IsItemHovered() && !a.error_detail.empty()) {
                ImGui::BeginTooltip();
                ImGui::PushTextWrapPos(px(620));
                ImGui::PushFont(nullptr, FS * 0.80f);
                ImGui::TextUnformatted(a.error_detail.c_str());
                ImGui::PopFont();
                ImGui::PopTextWrapPos();
                ImGui::EndTooltip();
            }
        }
    }
    ImGui::EndChild();

    // another engine is on the NPU: the user decides
    if (!a.busy_name.empty()) ImGui::OpenPopup("##busy");
    ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(px(26), px(22)));
    if (ImGui::BeginPopupModal("##busy", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
        ImGui::PushFont(g_semi, FS * 1.08f);
        ImGui::TextUnformatted(tr("Ett annat program använder NPU:n", "Another program is using the NPU"));
        ImGui::PopFont();
        ImGui::PushTextWrapPos(px(520));
        ImGui::Text(tr("%s är igång. Två körningar samtidigt kan störa varandra – vänta helst tills den är klar.",
                       "%s is running. Two runs at once can disturb each other – better wait until it is done."), a.busy_name.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Dummy(ImVec2(0, px(6)));
        if (primary_button(tr("Vänta", "Wait"), ImVec2(px(150), 0))) { a.busy_name.clear(); ImGui::CloseCurrentPopup(); }
        ImGui::SameLine();
        if (ImGui::Button(tr("Kör ändå", "Run anyway"), ImVec2(px(170), 0))) { a.busy_name.clear(); ImGui::CloseCurrentPopup(); if (a.busy_enlarge) a.start_enlarge(true); else a.start(true); }
        ImGui::EndPopup();
    }
    ImGui::PopStyleVar();
}

// ── right: the picture, what it was made from, the earlier ones ──
static void draw_right(App & a) {
    const ImGuiStyle & st = ImGui::GetStyle();
    begin_card("##view", ImVec2(0, 0), nullptr, 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
    const float strip = px(84), strip_h = strip + st.ScrollbarSize + px(8);
    const float info_h = 2 * ImGui::GetTextLineHeightWithSpacing() + ImGui::GetFrameHeightWithSpacing();
    const ImVec2 av = ImGui::GetContentRegionAvail();
    const float area_h = (std::max)(px(120), av.y - strip_h - info_h - 3 * st.ItemSpacing.y);
    ImDrawList * dl = ImGui::GetWindowDrawList();
    const ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("##picture", ImVec2(av.x, area_h));
    const bool have = a.view.srv && a.view_path == a.sel && !a.sel.empty();
    if (have) {
        const float k = (std::min)(av.x / a.view.w, area_h / a.view.h);
        const float w = a.view.w * k, h = a.view.h * k;
        const ImVec2 p(p0.x + (av.x - w) * 0.5f, p0.y + (area_h - h) * 0.5f);
        dl->AddImageRounded(a.view.ref(), p, ImVec2(p.x + w, p.y + h), ImVec2(0, 0), ImVec2(1, 1), IM_COL32_WHITE, px(14));
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) ShellExecuteW(nullptr, L"open", to_w(a.sel).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    } else {
        dl->AddRectFilled(p0, ImVec2(p0.x + av.x, p0.y + area_h), ImGui::GetColorU32(C_INPUT), px(14));
        const char * t = a.sel.empty() ? tr("Din bild visas här", "Your picture appears here") : tr("Hämtar bilden …", "Loading the picture …");
        const ImVec2 ts = ImGui::CalcTextSize(t);
        dl->AddText(ImVec2(p0.x + (av.x - ts.x) * 0.5f, p0.y + (area_h - ts.y) * 0.5f), ImGui::GetColorU32(C_DIM), t);
    }

    // what it was made from
    const auto it = a.meta.find(name_of(a.sel));
    const Meta * mt = it == a.meta.end() ? nullptr : &it->second;
    ImGui::BeginChild("##info", ImVec2(0, 2 * ImGui::GetTextLineHeightWithSpacing()), 0, ImGuiWindowFlags_NoScrollbar);
    if (mt) {
        std::string one = mt->prompt;
        std::replace(one.begin(), one.end(), '\n', ' ');
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextUnformatted(one.c_str());
        ImGui::PopTextWrapPos();
    } else if (!a.sel.empty()) ImGui::TextColored(C_DIM, "%s", a.sel.substr(a.sel.find_last_of('\\') + 1).c_str());
    ImGui::EndChild();
    if (mt && ImGui::IsItemHovered()) {
        char b[200];
        snprintf(b, sizeof b, tr("%d × %d · variant %u · %s · %.0f s · ungefär %.2f Wh", "%d × %d · variation %u · %s · %.0f s · about %.2f Wh"), mt->w, mt->h, mt->seed,
                 mt->model == "q8" ? tr("bäst", "best") : tr("snabbare", "faster"), mt->secs, watt_hours(mt->secs));
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(px(560));
        ImGui::TextUnformatted(mt->prompt.c_str());
        if (!mt->sent.empty()) ImGui::TextColored(C_DIM, "%s %s", tr("Skickat:", "Sent:"), mt->sent.c_str());
        ImGui::TextColored(C_DIM, "%s", b);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }

    ImGui::BeginDisabled(a.sel.empty());
    if (ImGui::Button(tr("Öppna", "Open"))) ShellExecuteW(nullptr, L"open", to_w(a.sel).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    ImGui::SameLine();
    if (ImGui::Button(tr("Visa i mappen", "Show in folder"))) {
        const std::wstring arg = L"/select,\"" + to_w(a.sel) + L"\"";
        ShellExecuteW(nullptr, L"open", L"explorer.exe", arg.c_str(), nullptr, SW_SHOWNORMAL);
    }
    ImGui::EndDisabled();
    ImGui::BeginDisabled(a.sel.empty() || a.running);
    if (a.can_edit()) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Ändra den här bilden", "Change this picture"))) { for (RefPic & r : a.refs) r.thumb.drop(); a.refs.clear(); a.add_ref(a.sel); a.prompt[0] = 0; }
    }
    if (mt) {
        ImGui::SameLine();
        if (ImGui::Button(tr("Återanvänd beskrivningen", "Reuse description"))) {
            snprintf(a.prompt, sizeof a.prompt, "%s", mt->prompt.c_str());
            snprintf(a.title, sizeof a.title, "%s", mt->title.c_str());
            snprintf(a.sub, sizeof a.sub, "%s", mt->sub.c_str());
            a.text_top = mt->top;
            a.seed = (int) mt->seed;
        }
    }
    if (a.enlarge_to()) {                                   // the picture that is shown, twice the size: a new file
        ImGui::SameLine();
        if (ImGui::Button(tr("Förstora 2\xC3\x97", "Enlarge 2\xC3\x97"))) a.start_enlarge();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip(tr("%d \xC3\x97 %d blir %d \xC3\x97 %d, sparas som en ny bild.", "%d \xC3\x97 %d becomes %d \xC3\x97 %d, saved as a new picture."), a.view_w, a.view_h, a.view_w * 2, a.view_h * 2);
    }
    if (a.title[0] || a.sub[0]) {                           // the text that is typed, on the picture that is shown: a new file, no NPU
        ImGui::SameLine();
        if (ImGui::Button(tr("Lägg på texten", "Add the text"))) a.start_text();
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("%s", tr("Skriver rubriken på bilden som visas och sparar den som en ny bild.", "Writes the title on the picture shown and saves it as a new picture."));
    }
    ImGui::EndDisabled();

    // the earlier pictures, newest first
    ImGui::BeginChild("##strip", ImVec2(0, strip_h), 0, ImGuiWindowFlags_HorizontalScrollbar);
    ImDrawList * sdl = ImGui::GetWindowDrawList();
    for (size_t i = 0; i < a.pics.size(); i++) {
        Pic & pc = a.pics[i];
        ImGui::PushID((int) i);
        const ImVec2 p = ImGui::GetCursorScreenPos();
        if (ImGui::InvisibleButton("##t", ImVec2(strip, strip))) a.select(pc.path);
        if (ImGui::IsItemVisible()) {
            if (pc.state == 0) { pc.state = 1; a.loader.push(LoadReq{ pc.path, 256, 1 }); }
            sdl->AddRectFilled(p, ImVec2(p.x + strip, p.y + strip), ImGui::GetColorU32(C_INPUT), px(12));
            if (pc.state == 2) draw_cover(sdl, pc.thumb, p, strip, px(12));
            if (pc.path == a.sel) sdl->AddRect(p, ImVec2(p.x + strip, p.y + strip), ImGui::GetColorU32(C_ACC_H), px(12), 0, px(3));
            else if (ImGui::IsItemHovered()) sdl->AddRect(p, ImVec2(p.x + strip, p.y + strip), ImGui::GetColorU32(ImVec4(1, 1, 1, 0.35f)), px(12), 0, px(2));
        }
        ImGui::PopID();
        if (i + 1 < a.pics.size()) ImGui::SameLine(0, px(10));
    }
    if (a.pics.empty()) hint(tr("Här samlas bilderna du gör.", "The pictures you make gather here."));
    ImGui::EndChild();
    end_card();
}

// Title bar: name, language, window buttons. Dragging is done by Windows (WM_NCHITTEST) outside the two control areas.
static void draw_titlebar(HWND hWnd, bool & done, App & a) {
    ImDrawList * dl = ImGui::GetWindowDrawList();
    const ImVec2 o = ImGui::GetWindowPos();
    const float W = ImGui::GetWindowSize().x, H = px(54);
    g_cap_h = (int) H;
    const float cy = o.y + H * 0.5f;
    dl->AddCircleFilled(ImVec2(o.x + px(30), cy), px(11), ImGui::GetColorU32(ImVec4(C_ACC.x, C_ACC.y, C_ACC.z, 0.28f)), 28);
    dl->AddCircleFilled(ImVec2(o.x + px(30), cy), px(6), ImGui::GetColorU32(C_ACC_H), 24);
    ImGui::PushFont(g_semi, FS * 1.05f);
    const float th = ImGui::GetTextLineHeight();
    ImGui::SetCursorScreenPos(ImVec2(o.x + px(50), cy - th * 0.5f));
    ImGui::TextUnformatted("PulseX Image");
    ImGui::PopFont();
    const float bw = px(46), bx = o.x + W - 3 * bw;
    // language
    const float ph = ImGui::GetFrameHeight(), lw_btn = px(104);
    const float lx = bx - lw_btn - px(14);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, ph * 0.5f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0));
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.06f));
    ImGui::PushStyleColor(ImGuiCol_Text, C_DIM);
    ImGui::SetCursorScreenPos(ImVec2(lx, cy - ph * 0.5f));
    if (ImGui::Button(g_lang ? "Svenska###lang" : "English###lang", ImVec2(lw_btn, 0))) { g_lang = !g_lang; a.settings_save(); }
    ImGui::PopStyleColor(3);
    ImGui::PopStyleVar(2);
    g_nodrag[0] = { (LONG) (lx - o.x), (LONG) (cy - ph * 0.5f - o.y), (LONG) (lx + lw_btn - o.x), (LONG) (cy + ph * 0.5f - o.y) };
    // window buttons
    g_nodrag[1] = { (LONG) (bx - o.x), 0, (LONG) W, (LONG) H };
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 0.0f);
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0, 0, 0, 0)); ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(1, 1, 1, 0.09f)); ImGui::PushStyleColor(ImGuiCol_ButtonActive, ImVec4(1, 1, 1, 0.16f));
    const ImU32 ic = ImGui::GetColorU32(C_TEXT);
    const float k = px(5), lw = px(1.2f);
    ImGui::SetCursorScreenPos(ImVec2(bx, o.y));
    if (ImGui::Button("##min", ImVec2(bw, H))) ShowWindow(hWnd, SW_MINIMIZE);
    { const ImVec2 c(bx + bw * 0.5f, cy); dl->AddLine(ImVec2(c.x - k, c.y), ImVec2(c.x + k, c.y), ic, lw); }
    ImGui::SetCursorScreenPos(ImVec2(bx + bw, o.y));
    {
        const bool mx = IsZoomed(hWnd) != 0;
        if (ImGui::Button("##max", ImVec2(bw, H))) ShowWindow(hWnd, mx ? SW_RESTORE : SW_MAXIMIZE);
        const ImVec2 c(bx + bw * 1.5f, cy);
        if (mx) {
            dl->AddRect(ImVec2(c.x - k + px(2), c.y - k), ImVec2(c.x + k, c.y + k - px(2)), ic, px(1.5f), 0, lw);
            dl->AddRectFilled(ImVec2(c.x - k, c.y - k + px(2)), ImVec2(c.x + k - px(2), c.y + k), ImGui::GetColorU32(C_BG), px(1.5f));
            dl->AddRect(ImVec2(c.x - k, c.y - k + px(2)), ImVec2(c.x + k - px(2), c.y + k), ic, px(1.5f), 0, lw);
        } else dl->AddRect(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), ic, px(1.5f), 0, lw);
    }
    ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.80f, 0.24f, 0.24f, 0.90f));
    ImGui::SetCursorScreenPos(ImVec2(bx + 2 * bw, o.y));
    if (ImGui::Button("##close", ImVec2(bw, H))) done = true;
    { const ImVec2 c(bx + bw * 2.5f, cy); dl->AddLine(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), ic, lw); dl->AddLine(ImVec2(c.x - k, c.y + k), ImVec2(c.x + k, c.y - k), ic, lw); }
    ImGui::PopStyleColor(4);
    ImGui::PopStyleVar(2);
}

// the window as a PNG (test hook --shot): the back buffer, before it is presented
static bool save_backbuffer(const std::string & path) {
    ID3D11Texture2D * back = nullptr, * stg = nullptr;
    if (FAILED(g_pSwapChain->GetBuffer(0, IID_PPV_ARGS(&back))) || !back) return false;
    D3D11_TEXTURE2D_DESC d;
    back->GetDesc(&d);
    d.Usage = D3D11_USAGE_STAGING; d.BindFlags = 0; d.CPUAccessFlags = D3D11_CPU_ACCESS_READ; d.MiscFlags = 0;
    bool ok = false;
    if (SUCCEEDED(g_pd3dDevice->CreateTexture2D(&d, nullptr, &stg)) && stg) {
        g_pd3dDeviceContext->CopyResource(stg, back);
        D3D11_MAPPED_SUBRESOURCE mp;
        if (SUCCEEDED(g_pd3dDeviceContext->Map(stg, 0, D3D11_MAP_READ, 0, &mp))) {
            fluximg::Rgb im;
            im.w = (int) d.Width; im.h = (int) d.Height;
            im.px.resize((size_t) im.w * im.h * 3);
            for (int y = 0; y < im.h; y++) {
                const uint8_t * s = (const uint8_t *) mp.pData + (size_t) y * mp.RowPitch;
                for (int x = 0; x < im.w; x++) { im.px[((size_t) y * im.w + x) * 3] = s[x * 4]; im.px[((size_t) y * im.w + x) * 3 + 1] = s[x * 4 + 1]; im.px[((size_t) y * im.w + x) * 3 + 2] = s[x * 4 + 2]; }
            }
            g_pd3dDeviceContext->Unmap(stg, 0);
            std::thread t([&]() { try { fluximg::save_png(path, im); ok = true; } catch (...) {} });      // WIC on a thread of its own
            t.join();
        }
        stg->Release();
    }
    back->Release();
    return ok;
}

// ════════════════════════ main ════════════════════════
int main(int argc, char ** argv) {
    std::string shot, demo, demo_size, demo_model, demo_title, demo_sub, demo_enrich, demo_format;
    bool demo_top = false, demo_addtext = false, demo_enlarge = false;
    int lang_arg = -1, shot_w = 0, shot_h = 0, demo_seed = -1, demo_up = -1;
    for (int i = 1; i < argc; i++) {
        const std::string s = argv[i];
        if (s == "--shot" && i + 1 < argc) shot = argv[++i];
        else if (s == "--demo-run" && i + 1 < argc) demo = argv[++i];
        else if (s == "--lang" && i + 1 < argc) lang_arg = std::string(argv[++i]) == "en" ? 1 : 0;
        else if (s == "--demo-ref" && i + 1 < argc) g_dropped.push_back(argv[++i]);          // as if the picture had been dropped on the window
        else if (s == "--demo-seed" && i + 1 < argc) demo_seed = atoi(argv[++i]);
        else if (s == "--demo-size" && i + 1 < argc) demo_size = argv[++i];                   // WxH, one of the shapes that exist
        else if (s == "--demo-model" && i + 1 < argc) demo_model = argv[++i];                 // q4 | q8
        else if (s == "--demo-up" && i + 1 < argc) demo_up = atoi(argv[++i]);                 // 0 | 1024 | 2048
        else if (s == "--demo-title" && i + 1 < argc) demo_title = argv[++i];
        else if (s == "--demo-sub" && i + 1 < argc) demo_sub = argv[++i];
        else if (s == "--demo-top") demo_top = true;
        else if (s == "--demo-enrich" && i + 1 < argc) demo_enrich = argv[++i];                 // off | nouns
        else if (s == "--demo-format" && i + 1 < argc) demo_format = argv[++i];                 // png | jpg
        else if (s == "--demo-addtext") demo_addtext = true;
        else if (s == "--demo-enlarge") demo_enlarge = true;                                  // press "Enlarge 2x" on the newest picture                                  // press "Add the text" on the newest picture (no NPU)
        else if (s == "--window" && i + 2 < argc) { shot_w = atoi(argv[++i]); shot_h = atoi(argv[++i]); }
    }
    WNDCLASSEXW wc{ sizeof(wc), CS_CLASSDC, WndProc, 0, 0, GetModuleHandle(nullptr), nullptr, LoadCursor(nullptr, IDC_ARROW), nullptr, nullptr, L"pulsex_image", nullptr };
    wc.hIcon = (HICON) LoadImageW(nullptr, to_w(app_dir() + "\\pulsex-image.ico").c_str(), IMAGE_ICON, 0, 0, LR_LOADFROMFILE | LR_DEFAULTSIZE);    // tools\make_icon_1002.py
    wc.hIconSm = (HICON) LoadImageW(nullptr, to_w(app_dir() + "\\pulsex-image.ico").c_str(), IMAGE_ICON, GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON), LR_LOADFROMFILE);
    RegisterClassExW(&wc);
    g_scale = GetDpiForSystem() / 96.0f;
    RECT wa{ 0, 0, 1280, 1024 };
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
    const int waW = wa.right - wa.left, waH = wa.bottom - wa.top;
    const int winW = shot_w ? shot_w : (std::min)((int) (1400 * g_scale), waW - (int) (24 * g_scale));
    const int winH = shot_h ? shot_h : (std::min)((int) (900 * g_scale), waH - (int) (24 * g_scale));
    const int winX = wa.left + (waW - winW) / 2, winY = wa.top + (waH - winH) / 2;
    HWND hWnd = CreateWindowExW(WS_EX_APPWINDOW | WS_EX_ACCEPTFILES, wc.lpszClassName, L"PulseX Image",
                                WS_POPUP | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU,   // no native caption
                                winX, winY, winW, winH, nullptr, nullptr, wc.hInstance, nullptr);
    {
        const DWORD round = 2 /*DWMWCP_ROUND*/;
        DwmSetWindowAttribute(hWnd, 33 /*DWMWA_WINDOW_CORNER_PREFERENCE*/, &round, sizeof(round));
        const BOOL dark = TRUE;
        DwmSetWindowAttribute(hWnd, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/, &dark, sizeof(dark));
        const COLORREF edge = RGB(84, 90, 101);
        DwmSetWindowAttribute(hWnd, 34 /*DWMWA_BORDER_COLOR*/, &edge, sizeof(edge));
    }
    g_scale = GetDpiForWindow(hWnd) / 96.0f;
    if (!CreateDeviceD3D(hWnd)) { CleanupDeviceD3D(); UnregisterClassW(wc.lpszClassName, wc.hInstance); return 1; }
    ShowWindow(hWnd, shot.empty() ? SW_SHOWDEFAULT : SW_SHOWNOACTIVATE);
    UpdateWindow(hWnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();
    io.IniFilename = nullptr;
    {
        std::string fonts = known_folder(FOLDERID_Fonts);
        if (fonts.empty()) fonts = "C:\\Windows\\Fonts";
        g_font = io.Fonts->AddFontFromFileTTF((fonts + "\\segoeui.ttf").c_str(), FS);
        if (!g_font) g_font = io.Fonts->AddFontDefault();
        g_semi = io.Fonts->AddFontFromFileTTF((fonts + "\\seguisb.ttf").c_str(), FS);
        if (!g_semi) g_semi = g_font;
    }
    apply_style();
    ImGui_ImplWin32_Init(hWnd);
    ImGui_ImplDX11_Init(g_pd3dDevice, g_pd3dDeviceContext);

    static App app;
    g_lang = 1;                                            // English first; Swedish is one click away
    app.find_things();
    app.settings_load();
    if (lang_arg >= 0) g_lang = lang_arg;
    app.find_things();                                     // again: the messages in the chosen language
    app.history_load();
    app.scan_pics();
    if (!app.pics.empty()) app.select(app.pics[0].path);
    app.loader.start();
    app.test_run = !shot.empty() || !demo.empty() || demo_addtext || demo_enlarge;
    if (!demo.empty()) snprintf(app.prompt, sizeof app.prompt, "%s", demo.c_str());
    if (demo_seed >= 0) { app.random_seed = false; app.seed = demo_seed; }
    for (size_t i = 0; i < app.sizes.size(); i++)
        if (std::to_string(app.sizes[i].w) + "x" + std::to_string(app.sizes[i].h) == demo_size) app.size_ix = (int) i;
    if ((demo_model == "q4" && app.has_q4) || (demo_model == "q8" && app.has_q8)) app.model = demo_model;
    snprintf(app.title, sizeof app.title, "%s", demo_title.c_str());
    snprintf(app.sub, sizeof app.sub, "%s", demo_sub.c_str());
    app.text_top = demo_top;
    if (!demo_enrich.empty()) app.swedish = demo_enrich != "off";
    if (!demo_format.empty()) app.png = demo_format == "png";
    if (demo_up >= 0) app.up = demo_up / (std::min)(app.size().w, app.size().h);       // --demo-up is the short side, as flux-make --up
    else if (app.test_run) app.up = 0;                     // a test run chooses everything itself

    bool done = false, quit_asked = false, demo_started = false;
    int frame = 0, frames_after = 0;
    int exit_code = 0;
    while (!done) {
        MSG msg;
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); if (msg.message == WM_QUIT) quit_asked = true; }
        // Throttle when minimised or not in focus: no busy loop. (A running picture still moves its bar.)
        if (IsIconic(hWnd) && shot.empty()) { Sleep(120); app.pump(); if (quit_asked && !app.running) break; continue; }
        if (GetForegroundWindow() != hWnd && shot.empty()) Sleep(30);
        if (g_new_dpi) { g_scale = g_new_dpi / 96.0f; g_new_dpi = 0; apply_style(); }
        app.pump();
        ImGui_ImplDX11_NewFrame();
        ImGui_ImplWin32_NewFrame();
        ImGui::NewFrame();
        const ImGuiViewport * vp = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(vp->Pos);
        ImGui::SetNextWindowSize(vp->Size);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::Begin("PulseX Image", nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse
                                               | ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar();
        bool close_clicked = false;
        draw_titlebar(hWnd, close_clicked, app);
        if (close_clicked) quit_asked = true;
        const float mg = px(18);
        ImGui::SetCursorPos(ImVec2(mg, (float) g_cap_h + px(2)));
        ImGui::BeginChild("##page", ImVec2(ImGui::GetWindowSize().x - 2 * mg, ImGui::GetWindowSize().y - g_cap_h - px(2) - mg), 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        {
            const ImVec2 av = ImGui::GetContentRegionAvail();
            const float leftw = (std::min)(px(470), av.x * 0.42f);
            ImGui::BeginChild("##left", ImVec2(leftw, av.y), 0, ImGuiWindowFlags_NoScrollbar);
            draw_left(app, hWnd);
            ImGui::EndChild();
            ImGui::SameLine(0, px(16));
            ImGui::BeginChild("##right", ImVec2(0, av.y), 0, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            draw_right(app);
            ImGui::EndChild();
        }
        ImGui::EndChild();
        if (quit_asked && app.running) {                   // never leave an engine behind: the window waits for the picture
            const char * t = tr("Avslutar när bilden är klar …", "Closing when the picture is done …");
            const ImVec2 ts = ImGui::CalcTextSize(t);
            ImGui::GetForegroundDrawList()->AddText(ImVec2(vp->Pos.x + px(200), vp->Pos.y + (g_cap_h - ts.y) * 0.5f), ImGui::GetColorU32(C_WARN), t);
        }
        ImGui::End();
        ImGui::Render();
        const float clr[4] = { C_BG.x, C_BG.y, C_BG.z, 1.0f };
        g_pd3dDeviceContext->OMSetRenderTargets(1, &g_mainRTV, nullptr);
        g_pd3dDeviceContext->ClearRenderTargetView(g_mainRTV, clr);
        ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

        // test hooks
        frame++;
        const bool demo_mode = !demo.empty() || demo_addtext || demo_enlarge;
        if (demo_mode && !demo_started && frame > 5 && (!demo_enlarge || app.view_path == app.sel)) {      // Enlarge needs the picture's size: wait for it
            demo_started = true;
            if (demo_enlarge) app.start_enlarge(true); else if (demo_addtext) app.start_text(); else app.start(true);
            if (!app.running) { fprintf(stderr, "demo: nothing started (%s)\n", app.not_ready.c_str()); exit_code = 3; quit_asked = true; }
        }
        if (!shot.empty()) {
            const bool settled = !demo_mode ? frame > 40 : (demo_started && !app.running && app.view_path == app.sel && ++frames_after > 20) || (demo_started && !app.running && !app.error.empty() && ++frames_after > 20);
            const bool mid = !demo.empty() && app.running && now_s() - app.t0 > 9.0 && frames_after == 0 && !file_exists(shot + ".mid.png");
            if (mid) save_backbuffer(shot + ".mid.png");
            if (settled) {
                if (!save_backbuffer(shot)) exit_code = 4;
                if (!app.error.empty()) { fprintf(stderr, "demo: %s\n%s\n", app.error.c_str(), app.error_detail.c_str()); exit_code = 5; }
                quit_asked = true;
            }
        }
        g_pSwapChain->Present(1, 0);
        if (quit_asked && !app.running) done = true;
    }
    app.settings_save();
    app.loader.shutdown();
    if (app.th.joinable()) app.th.join();
    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    CleanupDeviceD3D();
    DestroyWindow(hWnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);
    return exit_code;
}
