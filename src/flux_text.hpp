// flux_text.hpp - a title and a subtitle written on a finished picture (posters).
// 2026-10-02 (titles and subtitles are what makes a picture a poster). zimage.py's add_text() in C++:
// the same layout numbers - a margin of 5.5 % of the width, the title 7.5 % of the width and shrinking by 6 % at a
// time until it fits (not below 3 %), the subtitle 3 % (not below 1.8 %), a dark scrim that fades in over the band
// (one row at a time: steps show as bands against snow), white text with a soft black outline, left-aligned, at the
// bottom or the top. Why the text is put on afterwards and not asked of the model: a model draws texture that looks
// like text - a headline with a date comes out as nonsense. Composited text is spelled right, sharp, and can be
// changed without making the picture again.
// One thing zimage.py does not do: a long title is broken onto two lines rather than shrunk to a whisper.
// The text is drawn by GDI+ (outline = the glyphs' path stroked with a round pen, then filled), in Segoe UI Black,
// else Segoe UI Bold, else Arial Bold - the fonts zimage.py asks for, in its order.
#pragma once
#include "flux_image.hpp"

#include <objidl.h>
#ifdef NOMINMAX
namespace Gdiplus { using std::min; using std::max; }      // gdiplus.h uses the macros NOMINMAX takes away
#endif
#include <gdiplus.h>

#include <memory>

namespace fluxtext {

inline void add_text(fluximg::Rgb & im, const std::string & title, const std::string & subtitle, bool top) {
    if (title.empty() && subtitle.empty()) return;
    Gdiplus::GdiplusStartupInput gin;
    ULONG_PTR tok = 0;
    if (Gdiplus::GdiplusStartup(&tok, &gin, nullptr) != Gdiplus::Ok) throw std::runtime_error("GDI+ did not start");
    bool ok = true;
    {
        const int W = im.w, H = im.h;
        const int marg = (int) (W * 0.055);
        const float width = (float) (W - 2 * marg);
        const std::wstring wt = fluximg::wide(title), ws = fluximg::wide(subtitle);

        std::unique_ptr<Gdiplus::FontFamily> fam;
        INT style = Gdiplus::FontStyleRegular;
        const struct { const wchar_t * name; INT style; } want[] = { { L"Segoe UI Black", Gdiplus::FontStyleRegular }, { L"Segoe UI", Gdiplus::FontStyleBold }, { L"Arial", Gdiplus::FontStyleBold } };
        for (const auto & c : want) {
            fam.reset(new Gdiplus::FontFamily(c.name));
            style = c.style;
            if (fam->IsAvailable() && fam->IsStyleAvailable(style)) break;
            fam.reset();
        }
        if (!fam) { fam.reset(Gdiplus::FontFamily::GenericSansSerif()->Clone()); style = Gdiplus::FontStyleBold; }
        const Gdiplus::StringFormat * fmt = Gdiplus::StringFormat::GenericTypographic();      // no padding left of the first glyph

        auto ink_width = [&](const std::wstring & t, int px) {
            Gdiplus::GraphicsPath p;
            p.AddString(t.c_str(), -1, fam.get(), style, (Gdiplus::REAL) px, Gdiplus::PointF(0, 0), fmt);
            Gdiplus::RectF b;
            p.GetBounds(&b);
            return b.X + b.Width;
        };
        auto fit = [&](const std::wstring & t, int start_px, int min_px) {
            int px = start_px;
            while (px > min_px && ink_width(t, px) > width) px = (int) (px * 0.94);
            return px;
        };
        const int t_start = (std::max)(18, (int) (W * 0.075)), t_min = (std::max)(12, (int) (W * 0.030));
        int t_px = wt.empty() ? 0 : fit(wt, t_start, t_min);
        // A poster's headline (not in zimage.py): a title that would have to shrink below 70 % goes on TWO lines
        // instead, broken between the words where the longer line is shortest - if that lets it stay larger.
        std::vector<std::wstring> lines;
        if (!wt.empty()) lines.push_back(wt);
        if (!wt.empty() && t_px < t_start * 0.70) {
            size_t best = std::wstring::npos;
            float best_w = 1e30f;
            for (size_t i = wt.find(L' '); i != std::wstring::npos; i = wt.find(L' ', i + 1)) {
                if (i == 0 || i + 1 >= wt.size()) continue;
                const float w2 = (std::max)(ink_width(wt.substr(0, i), t_start), ink_width(wt.substr(i + 1), t_start));
                if (w2 < best_w) { best_w = w2; best = i; }
            }
            if (best != std::wstring::npos) {
                const std::wstring a = wt.substr(0, best), b = wt.substr(best + 1);
                const int px2 = (std::min)(fit(a, t_start, t_min), fit(b, t_start, t_min));
                if (px2 > t_px) { t_px = px2; lines = { a, b }; }
            }
        }
        const int t_step = (int) (t_px * 1.05);                           // from one title line to the next
        const int t_h = lines.empty() ? 0 : t_px + (int) (lines.size() - 1) * t_step;
        const int s_px = ws.empty() ? 0 : fit(ws, (std::max)(12, (int) (W * 0.030)), (std::max)(9, (int) (W * 0.018)));
        const int gap = wt.empty() ? 0 : (int) (t_px * 0.28);
        const int height = t_h + (!wt.empty() && !ws.empty() ? gap : 0) + s_px;
        const int band = (std::min)(H, height + 2 * marg);
        const int y0 = top ? 0 : H - band;

        // the scrim: (8, 12, 20) at an alpha that grows towards the picture's edge
        for (int k = 0; k < band; k++) {
            const double t = (double) k / (std::max)(1, band - 1);
            const int a = (int) (165.0 * pow(top ? 1.0 - t : t, 1.35));
            uint8_t * row = &im.px[(size_t) (y0 + k) * W * 3];
            for (int x = 0; x < W; x++, row += 3) {
                row[0] = (uint8_t) ((8 * a + row[0] * (255 - a) + 127) / 255);
                row[1] = (uint8_t) ((12 * a + row[1] * (255 - a) + 127) / 255);
                row[2] = (uint8_t) ((20 * a + row[2] * (255 - a) + 127) / 255);
            }
        }

        std::vector<uint8_t> buf((size_t) W * H * 4);                    // GDI+ draws on BGRA
        for (size_t i = 0, n = (size_t) W * H; i < n; i++) { buf[i * 4] = im.px[i * 3 + 2]; buf[i * 4 + 1] = im.px[i * 3 + 1]; buf[i * 4 + 2] = im.px[i * 3]; buf[i * 4 + 3] = 255; }
        {
            Gdiplus::Bitmap bmp(W, H, W * 4, PixelFormat32bppARGB, buf.data());
            Gdiplus::Graphics g(&bmp);
            g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
            g.SetPixelOffsetMode(Gdiplus::PixelOffsetModeHalf);
            auto draw = [&](const std::wstring & t, int px, int y, int stroke, BYTE stroke_a, Gdiplus::Color fill) {
                Gdiplus::GraphicsPath p;
                p.AddString(t.c_str(), -1, fam.get(), style, (Gdiplus::REAL) px, Gdiplus::PointF((Gdiplus::REAL) marg, (Gdiplus::REAL) y), fmt);
                Gdiplus::Pen pen(Gdiplus::Color(stroke_a, 0, 0, 0), (Gdiplus::REAL) (2 * stroke));      // the outline reaches `stroke` px outside the glyph
                pen.SetLineJoin(Gdiplus::LineJoinRound);
                Gdiplus::SolidBrush brush(fill);
                ok = ok && g.DrawPath(&pen, &p) == Gdiplus::Ok && g.FillPath(&brush, &p) == Gdiplus::Ok;
            };
            int y = y0 + marg;
            if (!wt.empty()) {
                for (size_t i = 0; i < lines.size(); i++) draw(lines[i], t_px, y + (int) i * t_step, (std::max)(1, t_px / 26), 190, Gdiplus::Color(255, 255, 255, 255));
                y += t_h + (ws.empty() ? 0 : gap);
            }
            if (!ws.empty()) draw(ws, s_px, y, (std::max)(1, s_px / 22), 170, Gdiplus::Color(255, 232, 238, 246));
            g.Flush(Gdiplus::FlushIntentionSync);
        }
        for (size_t i = 0, n = (size_t) W * H; i < n; i++) { im.px[i * 3] = buf[i * 4 + 2]; im.px[i * 3 + 1] = buf[i * 4 + 1]; im.px[i * 3 + 2] = buf[i * 4]; }
    }
    Gdiplus::GdiplusShutdown(tok);
    if (!ok) throw std::runtime_error("the text could not be drawn");
}

}  // namespace fluxtext
