// Bounded exact W/D/L classification for every legal action of a late UTTT root.
// Input batches: count, followed by [forced, own0, opp0, ..., own8, opp8].
// Output per root: visited nodes and 81 action classes, or '?' when incomplete.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Mask = uint16_t;
constexpr Mask full = 511;
constexpr std::array<Mask, 8> lines{7, 56, 448, 73, 146, 292, 273, 84};
std::array<bool, 512> wins{};
std::array<Mask, 512> winning_cells{};

void setup() {
    for (int mask = 0; mask <= full; ++mask) {
        for (Mask line : lines) wins[mask] |= (mask & line) == line;
    }
    for (int mask = 0; mask <= full; ++mask) {
        for (int c = 0; c < 9; ++c) {
            if (wins[mask | (1 << c)]) winning_cells[mask] |= 1 << c;
        }
    }
}

struct Position {
    Mask marks[2][9]{};
    Mask won[2]{};
    Mask open = full;
    int forced = 9;
    int turn = 0;
    uint64_t added = 0;
};

int pop_low(Mask& bits) {
    const int n = __builtin_ctz(static_cast<unsigned>(bits));
    bits &= bits - 1;
    return n;
}

struct Answer {
    uint64_t nodes = 0;
    std::string classes;
};

struct Budget {};
enum Bound : uint8_t { exact, lower, upper };
struct Entry {
    uint64_t key = 0;
    uint32_t root = 0;
    int8_t value = 0;
    Bound bound = exact;
};

class Solver {
    static constexpr size_t slots = 1 << 15;
    std::vector<Entry> table{slots};
    uint32_t root_id = 0;
    uint64_t nodes = 0;
    uint64_t limit;
    uint64_t added_bit[2][81]{};

    static Mask boards(const Position& p) {
        return p.forced == 9 ? p.open : static_cast<Mask>(1 << p.forced);
    }
    static Mask spaces(const Position& p, int board) {
        return full ^ (p.marks[0][board] | p.marks[1][board]);
    }
    Position child(const Position& p, int b, int c) const {
        Position next = p;
        next.marks[p.turn][b] |= 1 << c;
        bool won_here = wins[next.marks[p.turn][b]];
        if (won_here) next.won[p.turn] |= 1 << b;
        if (won_here || !spaces(next, b)) next.open &= ~(1 << b);
        next.forced = next.open & (1 << c) ? c : 9;
        next.turn ^= 1;
        next.added |= added_bit[p.turn][9*b + c];
        return next;
    }
    int search(const Position& p, int alpha, int beta) {
        if (nodes >= limit) throw Budget{};
        ++nodes;
        if (wins[p.won[1 - p.turn]]) return -1;
        // Even awarding every open board to one player could not make a line.
        if (!wins[p.won[0] | p.open] && !wins[p.won[1] | p.open]) return 0;

        // The root fixes all initial marks. Two bits per remaining square encode
        // every subsequent placement exactly. Closed boards are derived from marks.
        const uint64_t key = (p.added << 5) | (p.forced << 1) | p.turn;
        const auto slot = ((key ^ (key >> 29)) * 0x9E3779B97F4A7C15ULL) >> (64 - 15);
        Entry& entry = table[slot];
        if (entry.root == root_id && entry.key == key) {
            if (entry.bound == exact || (entry.bound == lower && entry.value >= beta)
                || (entry.bound == upper && entry.value <= alpha)) return entry.value;
        }

        int best = -1;
        const int original_alpha = alpha;
        Mask playable = boards(p);
        Mask local_wins[9]{}, blocks[9]{};
        for (Mask bs = playable; bs;) {
            const int b = pop_low(bs);
            const Mask empty = spaces(p, b);
            local_wins[b] = empty & winning_cells[p.marks[p.turn][b]];
            blocks[b] = empty & winning_cells[p.marks[1 - p.turn][b]] & ~local_wins[b];
            if (local_wins[b] && wins[p.won[p.turn] | (1 << b)]) {
                entry = {key, root_id, 1, exact};
                return 1;
            }
        }
        // Board captures, forced blocks, then other moves. Cut only on a valid
        // alpha-beta bound or the maximum possible value for the player.
        for (int priority = 0; priority < 3 && best < beta && best < 1; ++priority) {
            for (Mask bs = playable; bs && best < beta && best < 1;) {
                const int b = pop_low(bs);
                Mask moves = priority == 0 ? local_wins[b] : priority == 1 ? blocks[b]
                                    : spaces(p, b) & ~(local_wins[b] | blocks[b]);
                while (moves) {
                    const int c = pop_low(moves);
                    best = std::max(best, -search(child(p, b, c), -beta, -alpha));
                    alpha = std::max(alpha, best);
                    if (best >= beta || best == 1) break;
                }
            }
        }
        Bound bound = best >= beta ? lower : best <= original_alpha ? upper : exact;
        entry = {key, root_id, static_cast<int8_t>(best), bound};
        return best;
    }

public:
    explicit Solver(uint64_t cap) : limit(cap) {}
    Answer solve(const Position& root) {
        nodes = 0;
        if (++root_id == 0) {
            std::fill(table.begin(), table.end(), Entry{});
            root_id = 1;
        }
        int n = 0;
        for (Mask bs = root.open; bs;) {
            const int b = pop_low(bs);
            for (Mask cs = spaces(root, b); cs;) {
                const int c = pop_low(cs);
                if (n >= 28) throw std::runtime_error("more than 28 playable squares");
                added_bit[0][9*b+c] = 1ULL << (2*n);
                added_bit[1][9*b+c] = 1ULL << (2*n+1);
                ++n;
            }
        }
        std::string classes(81, '.');
        try {
            for (Mask bs = boards(root); bs;) {
                const int b = pop_low(bs);
                for (Mask cs = spaces(root, b); cs;) {
                    const int c = pop_low(cs);
                    // Every root action gets a full-window WDL search. A best-
                    // move-only root cutoff cannot establish the labeling mask.
                    classes[9*b+c] = "LDW"[1 - search(child(root, b, c), -2, 2)];
                }
            }
        } catch (const Budget&) {
            return {nodes, "?"};
        }
        return {nodes, classes};
    }
};

Position read_root() {
    Position root;
    if (!(std::cin >> root.forced) || root.forced < 0 || root.forced > 9)
        throw std::runtime_error("invalid forced board");
    int remaining = 0;
    for (int b = 0; b < 9; ++b) {
        int a = 0, o = 0;
        if (!(std::cin >> a >> o) || a < 0 || o < 0 || a > full || o > full || (a & o) || (wins[a] && wins[o]))
            throw std::runtime_error("invalid local board");
        root.marks[0][b] = a;
        root.marks[1][b] = o;
        if (wins[a]) root.won[0] |= 1 << b;
        if (wins[o]) root.won[1] |= 1 << b;
        if (wins[a] || wins[o] || (a | o) == full) root.open &= ~(1 << b);
        else remaining += 9 - __builtin_popcount(static_cast<unsigned>(a | o));
    }
    if (remaining > 28 || !root.open || wins[root.won[0]] || wins[root.won[1]]
        || (root.forced < 9 && !(root.open & (1 << root.forced))))
        throw std::runtime_error("expected a live endgame with at most 28 playable squares");
    return root;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: exact-targets nodes-per-root threads");
        const uint64_t cap = std::stoull(argv[1]);
        const int threads = std::stoi(argv[2]);
        if (!cap || threads < 1 || threads > 64) throw std::runtime_error("invalid limits");
        setup();
        std::vector<Solver> solvers;
        for (int i = 0; i < threads; ++i) solvers.emplace_back(cap);
        std::ios::sync_with_stdio(false);
        while (true) {
            size_t count;
            if (!(std::cin >> count)) {
                if (std::cin.eof()) return 0;
                throw std::runtime_error("invalid batch header");
            }
            if (!count || count > 4096) throw std::runtime_error("invalid batch size");
            std::vector<Position> roots;
            for (size_t i = 0; i < count; ++i) roots.push_back(read_root());
            std::vector<Answer> answers(count);
            std::atomic<size_t> next{0};
            auto work = [&](int thread) {
                size_t i;
                while ((i = next.fetch_add(1)) < count) answers[i] = solvers[thread].solve(roots[i]);
            };
            std::vector<std::thread> pool;
            for (int i = 1; i < threads && i < static_cast<int>(count); ++i) pool.emplace_back(work, i);
            work(0);
            for (auto& t : pool) t.join();
            for (const auto& a : answers) std::cout << a.nodes << ' ' << a.classes << '\n';
            std::cout.flush();
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
