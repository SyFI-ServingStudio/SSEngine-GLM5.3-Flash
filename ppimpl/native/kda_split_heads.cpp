// FlashInfer KDA persistent forward's `_split_heads` (flashinfer/kda_kernels/persistent/forward.py), the same search
// with the same tie-breaking, in C++: the Python one costs 20-35 ms per microbatch whose chains need splitting, on
// the forward's host path (ppimpl/model/kda_schedule.py).
#include <algorithm>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <vector>

namespace {
constexpr int PIECE_OVERHEAD = 3;
constexpr int MIN_PIECE = 2;
struct Job { int64_t cost; int kind, chain, c0, c1; };
}

extern "C" int kda_split_heads(int n, const int* chunks, const int* full, int nslots, double span_limit,
                               int64_t* out_span, int* out_slot, int* out_chain, int* out_c0, int* out_c1,
                               int* out_count) {
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return chunks[a] != chunks[b] ? chunks[a] > chunks[b] : a < b; });
    static const int NS[] = {8, 16, 24, 32, 48, 64, 96, 128, 192, 256};
    static const int HS[] = {2, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    bool found = false;
    int64_t best_span = 0;
    std::vector<std::vector<Job>> best_slots;
    for (int n_split : NS) {
        if (n_split > n) break;
        int room = full[order[0]];
        for (int i = 0; i < n_split; ++i) room = std::min(room, full[order[i]]);
        for (int head : HS) {
            if (head > room || chunks[order[n_split - 1]] - head < MIN_PIECE) break;
            std::vector<Job> jobs;
            jobs.reserve(n + n_split);
            for (int i = n_split; i < n; ++i) {
                int c = order[i];
                jobs.push_back({(int64_t)chunks[c] + PIECE_OVERHEAD, 1, c, 0, chunks[c]});
            }
            for (int i = 0; i < n_split; ++i) {
                int c = order[i];
                jobs.push_back({(int64_t)head + PIECE_OVERHEAD, 0, c, 0, head});
                jobs.push_back({(int64_t)chunks[c] - head + PIECE_OVERHEAD, 2, c, head, chunks[c]});
            }
            std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
                return std::make_tuple(-a.cost, a.kind, a.chain) < std::make_tuple(-b.cost, b.kind, b.chain); });
            using P = std::pair<int64_t, int>;
            std::priority_queue<P, std::vector<P>, std::greater<P>> heap;
            for (int s = 0; s < nslots; ++s) heap.push({0, s});
            std::vector<std::vector<Job>> slots(nslots);
            for (const Job& j : jobs) {
                P top = heap.top();
                heap.pop();
                slots[top.second].push_back(j);
                heap.push({top.first + j.cost, top.second});
            }
            std::vector<int64_t> ready(n, 0);
            for (auto& slot : slots) {
                std::stable_sort(slot.begin(), slot.end(), [](const Job& a, const Job& b) {
                    return std::make_pair(a.kind, -a.cost) < std::make_pair(b.kind, -b.cost); });
                int64_t t = 0;
                for (const Job& j : slot)
                    if (j.kind == 0) { t += j.cost; ready[j.chain] = t; }
            }
            int64_t span = 0;
            for (const auto& slot : slots) {
                int64_t t = 0;
                for (const Job& j : slot) {
                    if (j.kind == 2) t = std::max(t, ready[j.chain]);
                    t += j.cost;
                }
                span = std::max(span, t);
            }
            if ((double)span < span_limit && (!found || span < best_span)) {
                found = true;
                best_span = span;
                best_slots = slots;
            }
        }
    }
    if (!found) return 0;
    *out_span = best_span;
    int k = 0;
    for (int s = 0; s < (int)best_slots.size(); ++s)
        for (const Job& j : best_slots[s]) {
            out_slot[k] = s; out_chain[k] = j.chain; out_c0[k] = j.c0; out_c1[k] = j.c1; ++k;
        }
    *out_count = k;
    return 1;
}
