// flux_enrich.hpp - the Norrland theme: a prompt with Swedish words gets their English nouns, and (mode "full") the
// material words of zimage_dit's cue table.
// 2026-10-02 (zimage had a Norrland theme; FLUX should have it too). This is enrich.py of
// zimage_dit in C++, reading flux_cues.tsv (tools/make_cues_tsv_1002.py makes it from cues.json), with ONE
// difference that the pictures decided (tools/norrland_check_1002.py, norrland_moose_1002.py; seeds 4242 and 11, Q8):
//   * FLUX reads Swedish far better than Z-Image did - northern lights, snowy spruce forest, paper star and string
//     lights all arrive from the Swedish words alone. But "alg" came out as a deer 2 of 2 and "forsarna" as a calm
//     river 2 of 2.
//   * the English noun APPENDED (Z-Image's way: "en alg i ..., moose") still gave the deer 2 of 2: the Swedish word,
//     read first, wins. The English nouns FIRST ("moose, en alg i ...") gave a real moose 2 of 2, real rapids 2 of 2
//     and the cafe with its star 2 of 2.
// So here: the nouns go IN FRONT of the prompt, the material words (mode "full" only) after it.
// The material words are not needed for FLUX's realism and they move the camera (the moose became a close-up), so the
// default is "nouns": a prompt without Swedish key words is sent exactly as typed.
// Rules kept from enrich.py: whole words only (a trailing 's' allowed), the longest key wins over a key inside it,
// the nouns in the order their words appear in the prompt, fragments only while they fit the budget and never cut,
// nothing added that the prompt already says. Second difference: the nouns are not budgeted (the fragments are).
#pragma once
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace fluxenrich {

struct Table {
    int budget = 22;
    std::string photo_add;
    std::vector<std::string> photo_words;
    std::vector<std::pair<std::string, std::string>> front, cues, nouns;
    struct Guard { std::string key; std::vector<std::string> after, not_before; };
    std::vector<Guard> ambiguous;             // words that are only sometimes the thing ("ren": reindeer / clean)
    const Guard * guard(const std::string & k) const { for (const auto & g : ambiguous) if (g.key == k) return &g; return nullptr; }
    bool ok = false;
};

inline Table load(const std::string & path) {
    Table t;
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return t;
    std::string all;
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) all.append(b, n);
    fclose(f);
    size_t a = 0;
    while (a < all.size()) {
        size_t e = all.find('\n', a);
        if (e == std::string::npos) e = all.size();
        std::string line = all.substr(a, e - a);
        a = e + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.size() < 3 || line[0] == '#' || line[1] != '\t') continue;
        const size_t t2 = line.find('\t', 2);
        const std::string k = line.substr(2, t2 == std::string::npos ? std::string::npos : t2 - 2), v = t2 == std::string::npos ? "" : line.substr(t2 + 1);
        switch (line[0]) {
            case 'B': t.budget = atoi(k.c_str()); break;
            case 'A': t.photo_add = k; break;
            case 'P': t.photo_words.push_back(k); break;
            case 'F': t.front.push_back({ k, v }); break;
            case 'C': t.cues.push_back({ k, v }); break;
            case 'N': t.nouns.push_back({ k, v }); break;
            case 'G': {                       // key <tab> after words|not_before words
                Table::Guard g;
                g.key = k;
                const size_t bar = v.find('|');
                auto words = [](const std::string & w) { std::vector<std::string> o; size_t a = 0; while (a < w.size()) { size_t e = w.find(' ', a); if (e == std::string::npos) e = w.size(); if (e > a) o.push_back(w.substr(a, e - a)); a = e + 1; } return o; };
                g.after = words(v.substr(0, bar));
                if (bar != std::string::npos) g.not_before = words(v.substr(bar + 1));
                t.ambiguous.push_back(g);
                break;
            }
        }
    }
    t.ok = !t.nouns.empty() || !t.cues.empty();
    return t;
}

// lower case: ASCII, and the capital letters of Swedish (two bytes each in UTF-8: C3 85 / 84 / 96 / 89)
inline std::string lower(std::string s) {
    for (size_t i = 0; i < s.size(); i++) {
        const unsigned char c = (unsigned char) s[i];
        if (c >= 'A' && c <= 'Z') s[i] = (char) (c + 32);
        else if (c == 0xC3 && i + 1 < s.size()) {
            const unsigned char d = (unsigned char) s[i + 1];
            if (d >= 0x80 && d <= 0x9E && d != 0x97) s[i + 1] = (char) (d + 0x20);       // À..Þ -> à..þ (0x97 is the multiplication sign)
            i++;
        }
    }
    return s;
}
inline bool alpha(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80; }
inline bool alnum(unsigned char c) { return alpha(c) || (c >= '0' && c <= '9'); }
inline size_t chars(const std::string & s) { size_t n = 0; for (unsigned char c : s) if ((c & 0xC0) != 0x80) n++; return n; }

// the word that begins at i (forward) or ends at i (backward), after spaces; "" at a punctuation mark or the end
inline std::string word_at(const std::string & low, long i, int step) {
    const long n = (long) low.size();
    auto wc = [&](long k) { const unsigned char c = (unsigned char) low[(size_t) k]; return alpha(c) || c == '_' || c == '\''; };
    while (i >= 0 && i < n && low[(size_t) i] == ' ') i += step;
    long j = i;
    while (j >= 0 && j < n && wc(j)) j += step;
    return step > 0 ? low.substr((size_t) i, (size_t) (j - i)) : low.substr((size_t) (j + 1), (size_t) (i - j));
}
inline bool hasword(const std::string & low, const std::string & w, const Table::Guard * guard = nullptr) {
    if (w.empty()) return false;
    for (size_t p = low.find(w); p != std::string::npos; p = low.find(w, p + 1)) {
        if (p && (alpha((unsigned char) low[p - 1]) || low[p - 1] == '_')) continue;
        size_t e = p + w.size();
        if (e < low.size() && low[e] == 's') e++;
        if (e < low.size() && alpha((unsigned char) low[e])) continue;
        if (!guard) return true;
        // an ambiguous word counts only as a noun: the end, a punctuation mark or an "after" word follows it,
        // and no "not_before" word stands in front of it
        const std::string nxt = word_at(low, (long) e, 1), prv = word_at(low, (long) p - 1, -1);
        const bool after_ok = nxt.empty() || std::find(guard->after.begin(), guard->after.end(), nxt) != guard->after.end();
        const bool before_ok = std::find(guard->not_before.begin(), guard->not_before.end(), prv) == guard->not_before.end();
        if (after_ok && before_ok) return true;
    }
    return false;
}
inline int est_frag(const std::string & s) {
    int w = 0, p = 0;
    bool inw = false;
    for (unsigned char c : s) {
        const bool a = alnum(c) || c == '\'';
        if (a && !inw) { w++; inw = true; }
        else if (!a) inw = false;
        if (c == '-' || c == '/') p++;
    }
    return w + p + 1;
}
inline int est_prompt(const std::string & s) {
    int w = 0, p = 0;
    bool inw = false;
    for (unsigned char c : s) {
        const bool a = alnum(c);
        if (a && !inw) { w++; inw = true; }
        else if (!a) inw = false;
        if (c == '-' || c == '/' || c == '\'' || c == ',' || c == '.') p++;
    }
    return (std::max)(w + p, (int) ((s.size() + 2) / 3));
}
inline std::string trim(const std::string & s) {
    size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t')) b--;
    return s.substr(a, b - a);
}

// The prompt to send. full = false: only the English nouns of the Swedish words (in front); true: the material
// words too (after the prompt). `added` gets what was put in, in order. Unchanged prompt when nothing matches.
// nouns_first = false: the nouns are appended (before the material words) - Z-Image's order, as enrich.py does it.
inline std::string enrich(const Table & t, const std::string & raw, bool full, std::vector<std::string> * added = nullptr, bool nouns_first = true) {
    const std::string low = lower(raw);
    auto in = [](const std::vector<std::string> & v, const std::string & s) { return std::find(v.begin(), v.end(), s) != v.end(); };

    // the nouns: longest key first, a key inside an already matched longer key is skipped, then in the prompt's order
    std::vector<std::pair<std::string, std::string>> byLen = t.nouns;
    std::stable_sort(byLen.begin(), byLen.end(), [](const auto & a, const auto & b) { return chars(a.first) > chars(b.first); });
    std::vector<std::pair<std::string, std::string>> matched;
    for (const auto & kv : byLen) {
        if (!hasword(low, kv.first, t.guard(kv.first))) continue;
        bool inside = false;
        for (const auto & m : matched) if (m.first.find(kv.first) != std::string::npos) inside = true;
        if (!inside) matched.push_back(kv);
    }
    std::stable_sort(matched.begin(), matched.end(), [&](const auto & a, const auto & b) { return low.find(a.first) < low.find(b.first); });
    std::vector<std::string> nouns, frags;
    for (const auto & kv : matched)
        if (!hasword(low, lower(kv.second)) && !in(nouns, kv.second)) nouns.push_back(kv.second);

    if (full) {
        std::vector<std::pair<int, std::string>> hit;
        for (const auto & kv : t.front) if (hasword(low, kv.first, t.guard(kv.first))) hit.push_back({ (int) chars(kv.first) + 1000, kv.second });
        for (const auto & kv : t.cues) if (hasword(low, kv.first, t.guard(kv.first))) hit.push_back({ (int) chars(kv.first), kv.second });
        for (const auto & w : t.photo_words) if (hasword(low, w)) { hit.push_back({ 0, t.photo_add }); break; }
        std::stable_sort(hit.begin(), hit.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (const auto & h : hit) {
            size_t a = 0;
            for (;;) {
                const size_t e = h.second.find(',', a);
                const std::string frag = trim(h.second.substr(a, e == std::string::npos ? std::string::npos : e - a));
                if (!frag.empty() && low.find(lower(frag)) == std::string::npos && !in(nouns, frag) && !in(frags, frag)) frags.push_back(frag);
                if (e == std::string::npos) break;
                a = e + 1;
            }
        }
    }
    int budget = t.budget - est_prompt(raw);
    std::string head, tail;
    std::vector<std::string> used;
    for (const auto & a : nouns) {                 // the nouns ALWAYS go in (enrich.py budgets them: its budget was Z-Image's short
        head += a + ", ";                          // prompt; FLUX reads 512 tokens, and a long Swedish prompt got no help at all)
        budget -= est_frag(a);
        used.push_back(a);
    }
    for (const auto & a : frags) {
        const int k = est_frag(a);
        if (k > budget) continue;
        tail += ", " + a;
        budget -= k;
        used.push_back(a);
    }
    if (added) *added = used;
    if (used.empty()) return raw;
    std::string body = raw;
    if (!nouns_first && !head.empty()) {           // "a, b, " in front -> ", a, b" behind
        tail = ", " + head.substr(0, head.size() - 2) + tail;
        head.clear();
    }
    if (!tail.empty()) while (!body.empty() && (body.back() == ' ' || body.back() == ',' || body.back() == '.')) body.pop_back();
    return head + body + tail;
}

}  // namespace fluxenrich
