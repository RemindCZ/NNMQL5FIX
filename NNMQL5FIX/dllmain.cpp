// ============================================================================
//  dllmain.cpp — Lightweight MLP+LSTM DLL for MQL5 (x64, MSVC)
//  --------------------------------------------------------------------------
//  Max-precision build: stable sigmoid, precise dot products (Neumaier + FMA),
//  compensated sums for MSE & batch MSE, explicit type conversions.
//  API rozšířeno o trénink LSTM: BPTT/TBPTT, gradient clipping, SGD.
//  Paralelizace vypnuta (sekvenční).
//  (c) 2025, MIT-like spirit — use freely, please keep attribution.
//  NOTE (2025-09-03): MSE window set to ALWAYS-ON-TOP (WS_EX_TOPMOST + SetWindowPos).
// ============================================================================


// *** MUST be first because of PCH ***
#include "pch.h"
#include "NNMQL5_CUDA.cuh"   // prototypy CUDA exportů


#include <windows.h>
#include <vector>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <limits>
#include <cstdlib>
#include <algorithm>
#include <deque>
#include <atomic>
#include <random>

// ---------------------------------------------------------------------------
// Export / Calling convention
// ---------------------------------------------------------------------------
#ifndef DLL_EXTERN
#define DLL_EXTERN extern "C" __declspec(dllexport)
#endif
#ifndef DLL_CALL
#define DLL_CALL __cdecl
#endif

static HINSTANCE g_hInst = nullptr;

// ---------------------------------------------------------------------------
// RNG helper navázaný na NN_SetSeed (pro std::shuffle apod.)
// ---------------------------------------------------------------------------
namespace rng {
    static std::mt19937_64 g{ 1234567ULL };
    static std::mutex mtx;

    inline void seed(uint64_t s) {
        std::lock_guard<std::mutex> lk(mtx);
        g.seed(s);
    }
    template<typename It>
    inline void shuffle(It first, It last) {
        std::lock_guard<std::mutex> lk(mtx);
        std::shuffle(first, last, g);
    }
}

// ----------------------------------------------------------------------------
// Helper: bezpečný převod UTF-8 char* → std::wstring (UTF-16)
// ----------------------------------------------------------------------------
static std::wstring utf8_to_wide(const char* s) {
    if (!s) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return L"<invalid utf8>";
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}

// ----------------------------------------------------------------------------
// Precise numerics
// ----------------------------------------------------------------------------
namespace precise {
    inline double sigmoid(double x) {
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        else {
            const double z = std::exp(x);
            return z / (1.0 + z);
        }
    }

    inline double neumaier_sum_accumulate(double sum, double add, double& comp) {
        const double t = sum + add;
        if (std::abs(sum) >= std::abs(add)) {
            comp += (sum - t) + add;
        }
        else {
            comp += (add - t) + sum;
        }
        return t;
    }

    inline double dot_neumaier_fma(const double* a, const double* b, size_t n) {
        double sum = 0.0, c = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double prod = std::fma(a[i], b[i], 0.0);
            sum = neumaier_sum_accumulate(sum, prod, c);
        }
        return sum + c;
    }

    inline double sum_of_squares(const std::vector<double>& v) {
        double sum = 0.0, c = 0.0;
        for (double x : v) {
            const double sq = std::fma(x, x, 0.0);
            sum = neumaier_sum_accumulate(sum, sq, c);
        }
        return sum + c;
    }
} // namespace precise

// ----------------------------------------------------------------------------
// Tensor & Matrix (simple)
// ----------------------------------------------------------------------------
struct Tensor {
    size_t d{ 1 }, r{ 1 }, c{ 1 };
    std::vector<double> v;
    Tensor() = default;
    Tensor(size_t D, size_t R, size_t C, double init = 0.0)
        : d(D), r(R), c(C), v(D* R* C, init) {
    }
    inline size_t size() const { return v.size(); }
    inline double& at(size_t di, size_t ri, size_t ci) {
        return v[(di * r + ri) * c + ci];
    }
    inline const double& at(size_t di, size_t ri, size_t ci) const {
        return v[(di * r + ri) * c + ci];
    }
};

struct Matrix {
    size_t rows{ 0 }, cols{ 0 };
    std::vector<double> a;
    Matrix() = default;
    Matrix(size_t R, size_t C) : rows(R), cols(C), a(R* C) {}
    inline double& at(size_t r, size_t c) { return a[r * cols + c]; }
    inline const double& at(size_t r, size_t c) const { return a[r * cols + c]; }
};

// ----------------------------------------------------------------------------
// Activation
// ----------------------------------------------------------------------------
enum class ActKind : int { SIGMOID = 0, RELU = 1, TANH = 2, LINEAR = 3, SYM_SIG = 4 };

struct Activation {
    static double f(ActKind k, double x) {
        switch (k) {
        case ActKind::SIGMOID: return precise::sigmoid(x);
        case ActKind::RELU:    return x > 0.0 ? x : 0.0;
        case ActKind::TANH:    return std::tanh(x);
        case ActKind::SYM_SIG: return 2.0 * precise::sigmoid(x) - 1.0;
        case ActKind::LINEAR:  default: return x;
        }
    }
    static double df(ActKind k, double y, double x) {
        (void)x;
        switch (k) {
        case ActKind::SIGMOID: return y * (1.0 - y);
        case ActKind::RELU:    return x > 0.0 ? 1.0 : 0.0;
        case ActKind::TANH:    return 1.0 - y * y;
        case ActKind::SYM_SIG: return 0.5 * (1.0 - y * y);
        case ActKind::LINEAR:  default: return 1.0;
        }
    }
};

// ----------------------------------------------------------------------------
// DenseLayer
// ----------------------------------------------------------------------------
struct DenseLayer {
    size_t in_sz{ 0 }, out_sz{ 0 };
    Matrix W;
    std::vector<double> b;
    ActKind act{ ActKind::LINEAR };

    // caches for backprop
    std::vector<double> last_in, last_z, last_out;

    DenseLayer(size_t inSize, size_t outSize, ActKind k)
        : in_sz(inSize), out_sz(outSize), W(outSize, inSize), b(outSize, 0.0), act(k) {
        const double denom = (double)std::max<size_t>(1, in_sz);
        const double scale = (k == ActKind::RELU) ? std::sqrt(2.0 / denom)
            : std::sqrt(1.0 / denom);
        for (double& w : W.a) {
            const double u = (std::rand() / (double)RAND_MAX) * 2.0 - 1.0;
            w = u * scale;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) {
        if (x.size() != in_sz) {
            throw std::runtime_error("Dense forward: bad input size");
        }
        last_in = x;
        last_z.assign(out_sz, 0.0);
        last_out.assign(out_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            const double* wrow = &W.a[o * in_sz];
            const double  dot = precise::dot_neumaier_fma(wrow, x.data(), in_sz);
            const double  z = b[o] + dot;
            last_z[o] = z;
            last_out[o] = Activation::f(act, z);
        }
        return last_out;
    }

    std::vector<double> backward(const std::vector<double>& dL_dy, double lr) {
        std::vector<double> dL_dz(out_sz);
        for (size_t o = 0; o < out_sz; ++o) {
            const double y = last_out[o];
            const double z = last_z[o];
            dL_dz[o] = dL_dy[o] * Activation::df(act, y, z);
        }

        // gradient clipping (per-layer, conservative)
        const double gclip = 5.0;
        for (double& g : dL_dz) {
            if (g > gclip) {
                g = gclip;
            }
            else if (g < -gclip) {
                g = -gclip;
            }
        }

        // update b and W (FMA-friendly)
        for (size_t o = 0; o < out_sz; ++o) {
            b[o] -= lr * dL_dz[o];
            double* wrow = &W.a[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) {
                const double prod = std::fma(dL_dz[o], last_in[i], 0.0);
                wrow[i] = std::fma(-lr, prod, wrow[i]);
            }
        }

        // propagate gradient to inputs (compensated)
        std::vector<double> dL_dx(in_sz, 0.0), comp(in_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            const double* wrow = &W.a[o * in_sz];
            const double go = dL_dz[o];
            for (size_t i = 0; i < in_sz; ++i) {
                const double add = std::fma(wrow[i], go, 0.0);
                const double t = dL_dx[i] + add;
                if (std::abs(dL_dx[i]) >= std::abs(add)) {
                    comp[i] += (dL_dx[i] - t) + add;
                }
                else {
                    comp[i] += (add - t) + dL_dx[i];
                }
                dL_dx[i] = t;
            }
        }
        for (size_t i = 0; i < in_sz; ++i) {
            dL_dx[i] += comp[i];
        }
        return dL_dx;
    }
};

// ----------------------------------------------------------------------------
// MSELoss
// ----------------------------------------------------------------------------
struct MSELoss {
    static double loss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) {
            throw std::runtime_error("MSE size mismatch");
        }
        std::vector<double> e(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            e[i] = y[i] - t[i];
        }
        const double ss = precise::sum_of_squares(e);
        return ss / (double)y.size();
    }
    static std::vector<double> dloss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) {
            throw std::runtime_error("MSE size mismatch");
        }
        std::vector<double> g(y.size());
        const double n = (double)y.size();
        for (size_t i = 0; i < y.size(); ++i) {
            g[i] = std::fma(2.0, (y[i] - t[i]), 0.0) / n;
        }
        return g;
    }
};

// ----------------------------------------------------------------------------
// NeuralNetwork (Dense MLP)
// ----------------------------------------------------------------------------
class NeuralNetwork {
    std::vector<std::unique_ptr<DenseLayer>> layers;
    size_t input_size{ 0 }, output_size{ 0 };
public:
    bool add_dense(size_t in_sz, size_t out_sz, ActKind k) {
        if (layers.empty()) {
            input_size = in_sz;
        }
        else if (layers.back()->out_sz != in_sz) {
            return false;
        }
        layers.emplace_back(std::make_unique<DenseLayer>(in_sz, out_sz, k));
        output_size = out_sz;
        return true;
    }
    size_t in_size()  const { return input_size; }
    size_t out_size() const { return output_size; }

    bool forward(const double* in, int in_len, double* out, int out_len) {
        if ((int)input_size != in_len || (int)output_size != out_len || layers.empty()) {
            return false;
        }
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers) {
            x = L->forward(x);
        }
        for (int i = 0; i < out_len; ++i) {
            out[(size_t)i] = x[(size_t)i];
        }
        return true;
    }

    bool train_one(const double* in, int in_len, const double* tgt, int tgt_len,
        double lr, double* mse = nullptr) {
        if ((int)input_size != in_len || (int)output_size != tgt_len || layers.empty()) {
            return false;
        }
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers) {
            x = L->forward(x);
        }
        std::vector<double> t(tgt, tgt + tgt_len);
        if (mse) {
            *mse = MSELoss::loss(x, t);
        }
        std::vector<double> g = MSELoss::dloss(x, t);
        for (int li = (int)layers.size() - 1; li >= 0; --li) {
            g = layers[(size_t)li]->backward(g, lr);
        }
        return true;
    }

    bool get_weights(int i, double* W, int Wlen, double* b, int blen) const {
        if (i < 0) {
            return false;
        }
        const size_t u = (size_t)i;
        if (u >= layers.size()) {
            return false;
        }
        const DenseLayer* L = layers[u].get();
        if (!L) {
            return false;
        }
        const size_t needW = L->out_sz * L->in_sz;
        const size_t needb = L->out_sz;
        if ((size_t)Wlen != needW || (size_t)blen != needb) {
            return false;
        }
        std::memcpy(W, L->W.a.data(), needW * sizeof(double));
        std::memcpy(b, L->b.data(), needb * sizeof(double));
        return true;
    }

    bool set_weights(int i, const double* W, int Wlen, const double* b, int blen) {
        if (i < 0) {
            return false;
        }
        const size_t u = (size_t)i;
        if (u >= layers.size()) {
            return false;
        }
        DenseLayer* L = layers[u].get();
        if (!L) {
            return false;
        }
        const size_t needW = L->out_sz * L->in_sz;
        const size_t needb = L->out_sz;
        if ((size_t)Wlen != needW || (size_t)blen != needb) {
            return false;
        }
        std::memcpy(L->W.a.data(), W, needW * sizeof(double));
        std::memcpy(L->b.data(), b, needb * sizeof(double));
        return true;
    }
};

// ----------------------------------------------------------------------------
// LSTM (multilayer) — inference + training (SGD, TBPTT, clipping)
// ----------------------------------------------------------------------------
namespace lstm {

    inline void xavier_init(std::vector<double>& W, size_t fan_in, size_t fan_out) {
        const double denom = (double)std::max<size_t>(1, fan_in + fan_out);
        const double limit = std::sqrt(6.0 / denom);
        const auto randu = []() { return std::rand() / (double)RAND_MAX; };
        for (double& w : W) {
            const double u = randu() * 2.0 - 1.0;
            w = u * limit;
        }
    }

    struct LSTMLayer {
        size_t in_sz{ 0 }, hid_sz{ 0 };
        // Váhy v "concat" tvaru: [i|f|g|o]
        // W_x: (4*hid_sz) x in_sz, W_h: (4*hid_sz) x hid_sz, b: (4*hid_sz)
        std::vector<double> W_x, W_h, b;

        // Stav (aktuální, pro inference)
        std::vector<double> h, c;

        LSTMLayer(size_t inSize, size_t hidSize)
            : in_sz(inSize), hid_sz(hidSize),
            W_x(4 * hid_sz * in_sz), W_h(4 * hid_sz * hid_sz), b(4 * hid_sz, 0.0),
            h(hid_sz, 0.0), c(hid_sz, 0.0) {
        }

        void reset_state() {
            std::fill(h.begin(), h.end(), 0.0);
            std::fill(c.begin(), c.end(), 0.0);
        }

        void init_xavier() {
            xavier_init(W_x, in_sz, hid_sz);
            xavier_init(W_h, hid_sz, hid_sz);
            // Doporučení: forget bias lehce > 0 (lepší paměť)
            for (size_t r = 0; r < hid_sz; ++r) {
                b[hid_sz + r] = 1.0; // f-gate bias
            }
        }

        // Jeden krok: x -> (h,c) -> h (použito v online inference API)
        const std::vector<double>& step(const std::vector<double>& x) {
            const size_t HS = hid_sz;
            std::vector<double> z(4 * HS, 0.0);

            // z = W_x * x
            for (size_t g = 0; g < 4; ++g) {
                for (size_t r = 0; r < HS; ++r) {
                    const double* row = &W_x[(g * HS + r) * in_sz];
                    z[g * HS + r] = precise::dot_neumaier_fma(row, x.data(), in_sz);
                }
            }
            // + W_h * h + b
            for (size_t g = 0; g < 4; ++g) {
                for (size_t r = 0; r < HS; ++r) {
                    const double* row = &W_h[(g * HS + r) * HS];
                    const double add = precise::dot_neumaier_fma(row, h.data(), HS);
                    z[g * HS + r] += add + b[g * HS + r];
                }
            }

            // Brány
            for (size_t r = 0; r < HS; ++r) {
                const double i_t = precise::sigmoid(z[0 * HS + r]);
                const double f_t = precise::sigmoid(z[1 * HS + r]);
                const double g_t = std::tanh(z[2 * HS + r]);
                const double o_t = precise::sigmoid(z[3 * HS + r]);

                c[r] = std::fma(f_t, c[r], i_t * g_t);
                h[r] = o_t * std::tanh(c[r]);
            }
            return h;
        }
    };

    struct LSTMNet {
        size_t in_sz{ 0 }, hid_sz{ 0 }, out_sz{ 0 }, layers_n{ 0 };
        std::vector<LSTMLayer> layers;

        // Lineární projekce z posledního h na výstup
        std::vector<double> W_out; // out_sz x hid_sz
        std::vector<double> b_out; // out_sz

        // Tréninkové parametry
        int    tbptt_k = 0;         // 0 = plný BPTT
        double clip_norm = 0.0;     // 0 = bez clippingu

        LSTMNet(size_t inSize, size_t hidSize, size_t outSize, size_t nLayers)
            : in_sz(inSize), hid_sz(hidSize), out_sz(outSize), layers_n(nLayers),
            W_out(outSize* hidSize), b_out(outSize, 0.0) {
            layers.reserve(layers_n);
            size_t cur_in = in_sz;
            for (size_t i = 0; i < layers_n; ++i) {
                layers.emplace_back(cur_in, hid_sz);
                layers.back().init_xavier();
                cur_in = hid_sz;
            }
            xavier_init(W_out, hid_sz, out_sz);
        }

        void reset_state() {
            for (auto& L : layers) {
                L.reset_state();
            }
        }

        // Inference přes celou sekvenci: vrátí poslední výstup
        bool forward_last(const double* seq, int seq_len, double* out, int out_len) {
            if (!seq || !out || seq_len <= 0) {
                return false;
            }
            if ((int)out_sz != out_len) {
                return false;
            }

            std::vector<double> cur; cur.reserve(std::max(in_sz, hid_sz));
            for (int t = 0; t < seq_len; ++t) {
                cur.assign(seq + (size_t)t * in_sz, seq + (size_t)(t + 1) * in_sz);
                for (size_t li = 0; li < layers_n; ++li) {
                    cur = layers[li].step(cur);
                }
            }
            for (size_t o = 0; o < out_sz; ++o) {
                const double* row = &W_out[o * hid_sz];
                const double dot = precise::dot_neumaier_fma(row, cur.data(), hid_sz);
                out[o] = b_out[o] + dot;
            }
            return true;
        }

        // Inference: celá výstupní sekvence
        bool forward_seq(const double* seq, int seq_len, double* out, int out_len) {
            if (!seq || !out || seq_len <= 0) {
                return false;
            }
            if (out_len != seq_len * (int)out_sz) {
                return false;
            }

            std::vector<double> cur; cur.reserve(std::max(in_sz, hid_sz));
            for (int t = 0; t < seq_len; ++t) {
                cur.assign(seq + (size_t)t * in_sz, seq + (size_t)(t + 1) * in_sz);
                for (size_t li = 0; li < layers_n; ++li) {
                    cur = layers[li].step(cur);
                }
                double* yt = out + (size_t)t * out_sz;
                for (size_t o = 0; o < out_sz; ++o) {
                    const double* row = &W_out[o * hid_sz];
                    const double dot = precise::dot_neumaier_fma(row, cur.data(), hid_sz);
                    yt[o] = b_out[o] + dot;
                }
            }
            return true;
        }

        // ======== TRAIN ONE (SGD, TBPTT, clipping) ========
        bool train_one(const double* seq, int seq_len,
            const double* tgt, int tgt_len,
            double lr, double* mse_out)
        {
            if (!seq || !tgt || seq_len <= 0) {
                return false;
            }
            if ((int)out_sz != tgt_len) {
                return false;
            }

            const int T = seq_len;
            const size_t L = layers_n;
            const size_t H = hid_sz;
            const size_t IS = in_sz;
            const size_t OS = out_sz;

            // --- forward pass with caches ---
            // Store per-time, per-layer: h,c,i,f,g,o
            std::vector<std::vector<std::vector<double>>> h(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            std::vector<std::vector<std::vector<double>>> c(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            std::vector<std::vector<std::vector<double>>> gi(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            std::vector<std::vector<std::vector<double>>> gf(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            std::vector<std::vector<std::vector<double>>> gg(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            std::vector<std::vector<std::vector<double>>> go(L, std::vector<std::vector<double>>(T, std::vector<double>(H, 0.0)));
            // For layer-0 inputs (x_t)
            std::vector<std::vector<double>> x0(T, std::vector<double>(IS, 0.0));

            // temp vectors reused
            std::vector<double> cur_in, cur_out;
            cur_in.reserve(std::max(IS, H));
            cur_out.reserve(H);

            for (int t = 0; t < T; ++t) { // Hlavní časová smyčka
                // layer 0 input
                cur_in.assign(seq + (size_t)t * IS, seq + (size_t)(t + 1) * IS);
                x0[(size_t)t].assign(cur_in.begin(), cur_in.end());

                for (size_t li = 0; li < L; ++li) { // Smyčka přes vrstvy
                    LSTMLayer& Lr = layers[li];
                    const size_t HS = Lr.hid_sz;

                    // z = W_x * x + W_h * h_prev + b
                    std::vector<double> z(4 * HS, 0.0);

                    // W_x * x
                    const size_t inDim = (li == 0 ? IS : H);
                    for (size_t g = 0; g < 4; ++g) {
                        for (size_t r = 0; r < HS; ++r) {
                            const double* row = &Lr.W_x[(g * HS + r) * inDim];
                            z[g * HS + r] = precise::dot_neumaier_fma(row, cur_in.data(), inDim);
                        }
                    }
                    // + W_h * h + b
                    const std::vector<double> hprev = (t > 0 ? h[li][(size_t)(t - 1)] : std::vector<double>(HS, 0.0));
                    for (size_t g = 0; g < 4; ++g) {
                        for (size_t r = 0; r < HS; ++r) {
                            const double* row = &Lr.W_h[(g * HS + r) * HS];
                            z[g * HS + r] += precise::dot_neumaier_fma(row, hprev.data(), HS) + Lr.b[g * HS + r];
                        }
                    }

                    // gates
                    std::vector<double>& i_t = gi[li][(size_t)t];
                    std::vector<double>& f_t = gf[li][(size_t)t];
                    std::vector<double>& g_t = gg[li][(size_t)t];
                    std::vector<double>& o_t = go[li][(size_t)t];

                    for (size_t r = 0; r < HS; ++r) {
                        i_t[r] = precise::sigmoid(z[0 * HS + r]);
                        f_t[r] = precise::sigmoid(z[1 * HS + r]);
                        g_t[r] = std::tanh(z[2 * HS + r]);
                        o_t[r] = precise::sigmoid(z[3 * HS + r]);
                    }

                    // cell + hidden
                    std::vector<double>& c_t = c[li][(size_t)t];
                    std::vector<double>& h_t = h[li][(size_t)t];
                    for (size_t r = 0; r < HS; ++r) {
                        const double c_prev = (t > 0 ? c[li][(size_t)(t - 1)][r] : 0.0);
                        c_t[r] = std::fma(f_t[r], c_prev, i_t[r] * g_t[r]);
                        h_t[r] = o_t[r] * std::tanh(c_t[r]);
                    }

                    // prepare input for next layer
                    cur_in = h_t;
                }
            }

            // output at last time
            const std::vector<double>& h_last = h[L - 1][(size_t)(T - 1)];
            std::vector<double> y(OS, 0.0);
            for (size_t o = 0; o < OS; ++o) {
                const double* row = &W_out[o * H];
                const double dot = precise::dot_neumaier_fma(row, h_last.data(), H);
                y[o] = b_out[o] + dot;
            }

            // loss & grad at output
            std::vector<double> tvec(tgt, tgt + tgt_len);
            const double mse = MSELoss::loss(y, tvec);
            if (mse_out) {
                *mse_out = mse;
            }
            std::vector<double> dY = MSELoss::dloss(y, tvec);

            // grads accumulators
            std::vector<double> gW_out(OS * H, 0.0), gb_out(OS, 0.0);
            for (size_t o = 0; o < OS; ++o) {
                gb_out[o] += dY[o];
                for (size_t j = 0; j < H; ++j) {
                    gW_out[o * H + j] += dY[o] * h_last[j];
                }
            }
            // dh for top layer at T-1 from output projection
            std::vector<std::vector<double>> dh_next(L, std::vector<double>(H, 0.0));
            std::vector<std::vector<double>> dc_next(L, std::vector<double>(H, 0.0));
            // add W_out^T * dY to top layer dh at T-1
            for (size_t j = 0; j < H; ++j) {
                double s = 0.0, comp = 0.0;
                for (size_t o = 0; o < OS; ++o) {
                    const double add = std::fma(W_out[o * H + j], dY[o], 0.0);
                    const double tt = s + add;
                    if (std::abs(s) >= std::abs(add)) {
                        comp += (s - tt) + add;
                    }
                    else {
                        comp += (add - tt) + s;
                    }
                    s = tt;
                }
                dh_next[L - 1][j] = s + comp;
            }

            // param grads for LSTM layers
            struct LayerGrads {
                std::vector<double> dW_x; // (4H x in)
                std::vector<double> dW_h; // (4H x H)
                std::vector<double> db;   // (4H)
                LayerGrads(size_t in, size_t H)
                    : dW_x(4 * H * in, 0.0), dW_h(4 * H * H, 0.0), db(4 * H, 0.0) {
                }
            };
            std::vector<LayerGrads> grads;
            grads.reserve(L);
            for (size_t li = 0; li < L; ++li) {
                grads.emplace_back(li == 0 ? IS : H, H);
            }

            // ----- Backward Through Time -----
            const int k = (tbptt_k > 0 ? tbptt_k : T);
            const int t_start = std::max(0, T - k);

            for (int t = T - 1; t >= t_start; --t) { // Hlavní backprop časová smyčka
                // gradient "down" to lower layer at the same time-step
                std::vector<double> grad_down; // size depends on layer below
                for (int li = (int)L - 1; li >= 0; --li) { // Smyčka přes vrstvy
                    LSTMLayer& Lr = layers[(size_t)li];
                    LayerGrads& G = grads[(size_t)li];
                    const size_t inDim = (li == 0 ? IS : H);

                    // gather caches
                    const std::vector<double>& i_t = gi[(size_t)li][(size_t)t];
                    const std::vector<double>& f_t = gf[(size_t)li][(size_t)t];
                    const std::vector<double>& g_t = gg[(size_t)li][(size_t)t];
                    const std::vector<double>& o_t = go[(size_t)li][(size_t)t];
                    const std::vector<double>& c_t = c[(size_t)li][(size_t)t];
                    const std::vector<double>  c_prev = (t > 0 ? c[(size_t)li][(size_t)(t - 1)] : std::vector<double>(H, 0.0));
                    const std::vector<double>  h_prev = (t > 0 ? h[(size_t)li][(size_t)(t - 1)] : std::vector<double>(H, 0.0));
                    const std::vector<double>& x_t = (li == 0) ? x0[(size_t)t] : h[(size_t)(li - 1)][(size_t)t];

                    // incoming dh = from future time + from output (only top at last t) + from upper layer (grad_down)
                    std::vector<double> dh_cur = dh_next[(size_t)li];
                    if (li < (int)L - 1) {
                        if (grad_down.empty()) {
                            grad_down.assign(H, 0.0);
                        }
                        for (size_t j = 0; j < H; ++j) {
                            dh_cur[j] += grad_down[j];
                        }
                    }

                    // dc from future time
                    std::vector<double> dc_cur = dc_next[(size_t)li];

                    // do = dh * tanh(c_t) * o*(1-o)
                    std::vector<double> d_o(H, 0.0), d_i(H, 0.0), d_f(H, 0.0), d_g(H, 0.0);
                    for (size_t j = 0; j < H; ++j) {
                        const double tanh_c = std::tanh(c_t[j]);
                        const double do_pre = dh_cur[j] * tanh_c;
                        d_o[j] = do_pre * o_t[j] * (1.0 - o_t[j]);

                        // dc accumulates from dh through o & tanh
                        const double dc_from_dh = dh_cur[j] * o_t[j] * (1.0 - tanh_c * tanh_c);
                        const double dc_total = dc_cur[j] + dc_from_dh;

                        d_i[j] = dc_total * g_t[j] * i_t[j] * (1.0 - i_t[j]);
                        d_f[j] = dc_total * (t > 0 ? c_prev[j] : 0.0) * f_t[j] * (1.0 - f_t[j]);
                        d_g[j] = dc_total * i_t[j] * (1.0 - g_t[j] * g_t[j]);

                        // propagate to c_{t-1}
                        dc_next[(size_t)li][j] = dc_total * f_t[j];
                    }

                    // concat gates gradient
                    std::vector<double> d_concat(4 * H, 0.0);
                    for (size_t j = 0; j < H; ++j) {
                        d_concat[0 * H + j] = d_i[j];
                        d_concat[1 * H + j] = d_f[j];
                        d_concat[2 * H + j] = d_g[j];
                        d_concat[3 * H + j] = d_o[j];
                    }

                    // Accumulate parameter grads
                    // dW_x (4H x inDim)
                    for (size_t gk = 0; gk < 4; ++gk) {
                        for (size_t r = 0; r < H; ++r) {
                            const double val = d_concat[gk * H + r];
                            double* rowWx = &G.dW_x[(gk * H + r) * inDim];
                            for (size_t i = 0; i < inDim; ++i) {
                                rowWx[i] += val * x_t[i];
                            }
                        }
                    }
                    // dW_h (4H x H)
                    for (size_t gk = 0; gk < 4; ++gk) {
                        for (size_t r = 0; r < H; ++r) {
                            const double val = d_concat[gk * H + r];
                            double* rowWh = &G.dW_h[(gk * H + r) * H];
                            for (size_t j = 0; j < H; ++j) {
                                rowWh[j] += val * (t > 0 ? h_prev[j] : 0.0);
                            }
                        }
                    }
                    // db (4H)
                    for (size_t idx = 0; idx < 4 * H; ++idx) {
                        G.db[idx] += d_concat[idx];
                    }

                    // compute dx for lower layer (or input grad)
                    std::vector<double> dx(inDim, 0.0);
                    // dx += W_x^T * d_concat
                    for (size_t gk = 0; gk < 4; ++gk) {
                        for (size_t r = 0; r < H; ++r) {
                            const double val = d_concat[gk * H + r];
                            const double* rowWx = &Lr.W_x[(gk * H + r) * inDim];
                            for (size_t i = 0; i < inDim; ++i) {
                                dx[i] += rowWx[i] * val;
                            }
                        }
                    }
                    // dh_prev = W_h^T * d_concat
                    std::vector<double> dh_prev(H, 0.0);
                    for (size_t gk = 0; gk < 4; ++gk) {
                        for (size_t r = 0; r < H; ++r) {
                            const double val = d_concat[gk * H + r];
                            const double* rowWh = &Lr.W_h[(gk * H + r) * H];
                            for (size_t j = 0; j < H; ++j) {
                                dh_prev[j] += rowWh[j] * val;
                            }
                        }
                    }
                    dh_next[(size_t)li] = dh_prev;

                    // propagate dx to lower layer as its dh contribution at same time t
                    grad_down = dx;
                } // end layer loop
            } // end time loop

            // ---- Gradient clipping (global L2 over all params) ----
            double gsum = 0.0, gcomp = 0.0;
            auto acc_sq = [&](double v) {
                const double sq = std::fma(v, v, 0.0);
                const double t = gsum + sq;
                if (std::abs(gsum) >= std::abs(sq)) {
                    gcomp += (gsum - t) + sq;
                }
                else {
                    gcomp += (sq - t) + gsum;
                }
                gsum = t;
                };
            for (double v : gW_out) acc_sq(v);
            for (double v : gb_out) acc_sq(v);
            for (size_t li = 0; li < L; ++li) {
                for (double v : grads[li].dW_x) acc_sq(v);
                for (double v : grads[li].dW_h) acc_sq(v);
                for (double v : grads[li].db)   acc_sq(v);
            }
            const double gnorm = std::sqrt(gsum + gcomp);
            double scale = 1.0;
            if (clip_norm > 0.0 && gnorm > clip_norm) {
                scale = clip_norm / gnorm;
            }

            // ---- SGD update (with scaling if clipped) ----
            const double eta = lr;
            for (size_t o = 0; o < OS; ++o) {
                b_out[o] = std::fma(-eta * scale, gb_out[o], b_out[o]);
                for (size_t j = 0; j < H; ++j) {
                    const double g = gW_out[o * H + j] * scale;
                    W_out[o * H + j] = std::fma(-eta, g, W_out[o * H + j]);
                }
            }
            for (size_t li = 0; li < L; ++li) {
                LSTMLayer& Lr = layers[li];
                LayerGrads& G = grads[li];
                // W_x
                for (size_t idx = 0; idx < G.dW_x.size(); ++idx) {
                    Lr.W_x[idx] = std::fma(-eta, G.dW_x[idx] * scale, Lr.W_x[idx]);
                }
                // W_h
                for (size_t idx = 0; idx < G.dW_h.size(); ++idx) {
                    Lr.W_h[idx] = std::fma(-eta, G.dW_h[idx] * scale, Lr.W_h[idx]);
                }
                // b
                for (size_t idx = 0; idx < G.db.size(); ++idx) {
                    Lr.b[idx] = std::fma(-eta * scale, G.db[idx], Lr.b[idx]);
                }
            }

            return true;
        }

        // ======== TRAIN BATCH (sekvenčně přes vzorky) ========
        bool train_batch(const double* seq_batch, int batch, int seq_len,
            const double* tgt_batch, int tgt_len,
            double lr, double* mean_mse)
        {
            if (batch <= 0) {
                return false;
            }
            double acc = 0.0, comp = 0.0; int cnt = 0;
            for (int b = 0; b < batch; ++b) {
                const double* xi = seq_batch + (size_t)b * seq_len * in_sz;
                const double* ti = tgt_batch + (size_t)b * tgt_len;
                double mse = 0.0;
                if (!train_one(xi, seq_len, ti, tgt_len, lr, &mse)) {
                    return false;
                }
                if (std::isfinite(mse)) {
                    const double t = acc + mse;
                    if (std::abs(acc) >= std::abs(mse)) {
                        comp += (acc - t) + mse;
                    }
                    else {
                        comp += (mse - t) + acc;
                    }
                    acc = t; ++cnt;
                }
            }
            if (mean_mse) {
                const double sum = acc + comp;
                *mean_mse = (cnt > 0 ? sum / (double)cnt : 0.0);
            }
            return true;
        }
    };

    // ----------------------------------------------------------------------------
    // Instance management
    // ----------------------------------------------------------------------------
    static std::unordered_map<int, std::unique_ptr<LSTMNet>> g_lstm;
    static std::mutex g_lstm_mtx;
    static int g_lstm_next = 100000; // oddělený prostor handlů

    static int  lstm_alloc(size_t in_sz, size_t hid_sz, size_t out_sz, size_t layers_n) {
        std::lock_guard<std::mutex> lk(g_lstm_mtx);
        const int h = g_lstm_next++;
        g_lstm.emplace(h, std::make_unique<LSTMNet>(in_sz, hid_sz, out_sz, layers_n));
        return h;
    }
    static LSTMNet* lstm_get(int h) {
        std::lock_guard<std::mutex> lk(g_lstm_mtx);
        auto it = g_lstm.find(h);
        return it == g_lstm.end() ? nullptr : it->second.get();
    }
    static void lstm_free(int h) {
        std::lock_guard<std::mutex> lk(g_lstm_mtx);
        g_lstm.erase(h);
    }
} // namespace lstm

// ----------------------------------------------------------------------------
// Lightweight MSE monitor window (optional)
// ----------------------------------------------------------------------------
namespace ui {
    struct MSEPlotState {
        std::atomic<bool> running{ false };
        std::atomic<bool> visible{ false };
        HANDLE thread{ nullptr };
        DWORD  tid{ 0 };
        HWND   hwnd{ nullptr };
        std::mutex mtx;
        std::deque<double> data;
        size_t max_points = 1000;
        bool   autoscale = true;
        double y_min = 0.0, y_max = 1.0;
    };
    static MSEPlotState g;

    static const wchar_t* kClass = L"NNMSEWindowClass";
    static const wchar_t* kTitle = L"Remind_NNMQL5FIX — MSE";

    static void DrawMSEGraph(HDC hdc, RECT rc) {
        // background
        HBRUSH bg = (HBRUSH)(COLOR_WINDOW + 1);
        FillRect(hdc, &rc, bg);

        std::deque<double> local;
        {
            std::lock_guard<std::mutex> lk(g.mtx);
            local = g.data;
        }
        if (local.empty()) {
            return;
        }

        // compute bounds
        double vmin = std::numeric_limits<double>::infinity();
        double vmax = -std::numeric_limits<double>::infinity();
        if (g.autoscale) {
            for (double v : local) {
                if (std::isfinite(v)) {
                    if (v < vmin) vmin = v;
                    if (v > vmax) vmax = v;
                }
            }
            if (!std::isfinite(vmin) || !std::isfinite(vmax) || vmin == vmax) {
                vmin = 0.0; vmax = 1.0;
            }
        }
        else {
            vmin = g.y_min; vmax = g.y_max;
            if (vmin == vmax) {
                vmax = vmin + 1.0;
            }
        }

        // plot area margins
        const int L = 48, T = 8, R = 8, B = 28;
        RECT pr = { rc.left + L, rc.top + T, rc.right - R, rc.bottom - B };
        if (pr.right - pr.left < 4 || pr.bottom - pr.top < 4) {
            return;
        }

        // axes
        HPEN gridPen = CreatePen(PS_DOT, 1, RGB(200, 200, 200));
        HPEN axPen = CreatePen(PS_SOLID, 1, RGB(120, 120, 120));
        HPEN lnPen = CreatePen(PS_SOLID, 2, RGB(0, 120, 215));
        HFONT font = (HFONT)GetStockObject(DEFAULT_GUI_FONT);

        HGDIOBJ oldPen = SelectObject(hdc, axPen);
        HGDIOBJ oldFont = SelectObject(hdc, font);

        // frame
        Rectangle(hdc, pr.left, pr.top, pr.right, pr.bottom);

        // grid
        SelectObject(hdc, gridPen);
        for (int i = 1; i <= 3; ++i) {
            int y = pr.top + (pr.bottom - pr.top) * i / 4;
            MoveToEx(hdc, pr.left, y, nullptr);
            LineTo(hdc, pr.right, y);
        }

        // Y ticks text
        wchar_t buf[64];
        SetBkMode(hdc, TRANSPARENT);
        SetTextColor(hdc, RGB(80, 80, 80));
        for (int i = 0; i <= 4; ++i) {
            double val = vmin + (vmax - vmin) * i / 4.0;
            int y = pr.bottom - (int)std::lround((val - vmin) * (pr.bottom - pr.top) / (vmax - vmin));
            swprintf_s(buf, L"%.4g", val);
            TextOutW(hdc, rc.left + 2, y - 8, buf, (int)wcslen(buf));
        }

        // polyline
        SelectObject(hdc, lnPen);
        const int N = (int)local.size();
        const int W = pr.right - pr.left;
        const int H = pr.bottom - pr.top;

        std::vector<POINT> pts; pts.reserve(N);
        for (int i = 0; i < N; ++i) {
            const double v = local[i];
            const double xf = (N > 1 ? (double)i / (double)(N - 1) : 0.0);
            const int x = pr.left + (int)std::lround(xf * W);
            int y = pr.bottom - (int)std::lround((v - vmin) * H / (vmax - vmin));
            if (y < pr.top)   y = pr.top;
            if (y > pr.bottom) y = pr.bottom;
            pts.push_back(POINT{ x, y });
        }
        if (pts.size() >= 2) {
            Polyline(hdc, pts.data(), (int)pts.size());
        }

        // last value
        double last = local.back();
        swprintf_s(buf, L"MSE: %.6g", last);
        TextOutW(hdc, pr.left + 4, pr.bottom + 6, buf, (int)wcslen(buf));

        // cleanup
        SelectObject(hdc, oldPen);
        SelectObject(hdc, oldFont);
        DeleteObject(gridPen);
        DeleteObject(axPen);
        DeleteObject(lnPen);
    }

    static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
        switch (msg) {
        case WM_CREATE: {
            g.hwnd = hwnd;
            SetTimer(hwnd, 1, 1000 / 20, NULL); // ~20 FPS invalidation
            return 0;
        }
        case WM_TIMER: {
            InvalidateRect(hwnd, NULL, FALSE);
            return 0;
        }
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(hwnd, &ps);
            RECT rc; GetClientRect(hwnd, &rc);
            DrawMSEGraph(hdc, rc);
            EndPaint(hwnd, &ps);
            return 0;
        }
        case WM_CLOSE: {
            g.visible = false;
            ShowWindow(hwnd, SW_HIDE);
            return 0;
        }
        case WM_DESTROY: {
            KillTimer(hwnd, 1);
            PostQuitMessage(0);
            return 0;
        }
        }
        return DefWindowProc(hwnd, msg, wParam, lParam);
    }

    static DWORD WINAPI ThreadProc(LPVOID) {
        WNDCLASSW wc = {};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = g_hInst;
        wc.lpszClassName = kClass;
        wc.hCursor = LoadCursor(NULL, IDC_ARROW);
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
        RegisterClassW(&wc);

        const int w = 520, h = 260;
        HWND hwnd = CreateWindowExW(
            WS_EX_TOOLWINDOW | WS_EX_TOPMOST, // ALWAYS-ON-TOP extended style
            kClass, kTitle,
            WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_MINIMIZEBOX,
            CW_USEDEFAULT, CW_USEDEFAULT, w, h,
            nullptr, nullptr, g_hInst, nullptr);

        g.hwnd = hwnd;

        // Pin on top even if some shells ignore WS_EX_TOPMOST at creation
        SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE);

        ShowWindow(hwnd, g.visible ? SW_SHOW : SW_HIDE);
        UpdateWindow(hwnd);

        MSG msg;
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        g.hwnd = nullptr;
        return 0;
    }

    static void ensure_thread() {
        if (g.running.load()) {
            return;
        }
        g.visible = true;
        g.running = true;
        DWORD tid = 0;
        HANDLE th = CreateThread(nullptr, 0, ThreadProc, nullptr, 0, &tid);
        g.thread = th; g.tid = tid;
    }

    static void shutdown() {
        if (!g.running.load()) {
            return;
        }
        if (g.hwnd) {
            PostMessageW(g.hwnd, WM_DESTROY, 0, 0);
        }
        if (g.tid) {
            PostThreadMessageW(g.tid, WM_QUIT, 0, 0);
        }
        if (g.thread) {
            WaitForSingleObject(g.thread, 1000);
            CloseHandle(g.thread);
        }
        g.thread = nullptr; g.tid = 0; g.hwnd = nullptr;
        g.running = false; g.visible = false;
    }
}

// ----------------------------------------------------------------------------
// Instance management — Dense MLP
// ----------------------------------------------------------------------------
static std::unordered_map<int, std::unique_ptr<NeuralNetwork>> g_nets;
static std::mutex g_mtx;
static int g_next_handle = 1;

static int  alloc_handle() {
    std::lock_guard<std::mutex> lk(g_mtx);
    const int h = g_next_handle++;
    g_nets.emplace(h, std::make_unique<NeuralNetwork>());
    return h;
}
static NeuralNetwork* get_net(int h) {
    std::lock_guard<std::mutex> lk(g_mtx);
    auto it = g_nets.find(h);
    return it == g_nets.end() ? nullptr : it->second.get();
}
static void free_handle(int h) {
    std::lock_guard<std::mutex> lk(g_mtx);
    g_nets.erase(h);
}

// ============================================================================
// Error & Cancel state per handle
// ============================================================================
namespace nnctl {
    struct LastError { int code{ 0 }; std::wstring msg; };
    static std::unordered_map<int, LastError> g_err_nn, g_err_lstm;
    static std::unordered_map<int, std::atomic<int>> g_cancel_nn, g_cancel_lstm;
    static std::mutex g_err_mtx;

    inline void set_err(std::unordered_map<int, LastError>& m, int h, int code, const wchar_t* wmsg) {
        std::lock_guard<std::mutex> lk(g_err_mtx);
        LastError& e = m[h];
        e.code = code; e.msg = wmsg ? wmsg : L"";
    }
    inline void clear_err(std::unordered_map<int, LastError>& m, int h) {
        std::lock_guard<std::mutex> lk(g_err_mtx);
        auto it = m.find(h);
        if (it != m.end()) {
            it->second.code = 0; it->second.msg.clear();
        }
    }
}


// ============================================================================
// Forward declarations for DLL_EXTERN functions used before their definitions
// ============================================================================
DLL_EXTERN void  DLL_CALL NN_MSE_Push(double mse);
DLL_EXTERN int   DLL_CALL LSTM_SetTBPTT(int h, int k_steps);
DLL_EXTERN bool  DLL_CALL LSTM_TrainBatch(int h, const double* seq_batch, int batch, int seq_len,
    const double* tgt_batch, int tgt_len, double lr, double* mean_mse);


// ----------------------------------------------------------------------------
// Exported C API (cdecl) — core NN, matches MQL5 imports
// ----------------------------------------------------------------------------
DLL_EXTERN int   DLL_CALL NN_Create() { return alloc_handle(); }
DLL_EXTERN void  DLL_CALL NN_Free(int h) { free_handle(h); }

DLL_EXTERN bool  DLL_CALL NN_AddDense(int h, int inSz, int outSz, int act) {
    NeuralNetwork* net = get_net(h);
    if (!net) {
        return false;
    }
    ActKind k = (act == 0 ? ActKind::SIGMOID
        : act == 1 ? ActKind::RELU
        : act == 2 ? ActKind::TANH
        : act == 4 ? ActKind::SYM_SIG
        : ActKind::LINEAR);
    return net->add_dense((size_t)inSz, (size_t)outSz, k);
}

DLL_EXTERN int   DLL_CALL NN_InputSize(int h) { auto* n = get_net(h); return n ? (int)n->in_size() : 0; }
DLL_EXTERN int   DLL_CALL NN_OutputSize(int h) { auto* n = get_net(h); return n ? (int)n->out_size() : 0; }

DLL_EXTERN bool  DLL_CALL NN_Forward(int h, const double* in, int in_len, double* out, int out_len) {
    auto* n = get_net(h); return n ? n->forward(in, in_len, out, out_len) : false;
}

DLL_EXTERN bool  DLL_CALL NN_TrainOne(int h, const double* in, int in_len,
    const double* tgt, int tgt_len,
    double lr, double* mse) {
    auto* n = get_net(h); return n ? n->train_one(in, in_len, tgt, tgt_len, lr, mse) : false;
}

DLL_EXTERN bool  DLL_CALL NN_ForwardBatch(int h, const double* in, int batch, int in_len,
    double* out, int out_len) {
    if (batch <= 0) {
        return false;
    }
    NeuralNetwork* n = get_net(h);
    if (!n) {
        return false;
    }
    if (n->in_size() != (size_t)in_len || n->out_size() != (size_t)out_len) {
        return false;
    }

    for (int b = 0; b < batch; ++b) {
        const double* xi = in + (size_t)b * in_len;
        double* yi = out + (size_t)b * out_len;
        if (!n->forward(xi, in_len, yi, out_len)) {
            return false;
        }
    }
    return true;
}

DLL_EXTERN bool  DLL_CALL NN_TrainBatch(int h, const double* in, int batch, int in_len,
    const double* tgt, int tgt_len,
    double lr, double* mean_mse) {
    if (batch <= 0) {
        return false;
    }
    NeuralNetwork* n = get_net(h);
    if (!n) {
        return false;
    }
    if (n->in_size() != (size_t)in_len || n->out_size() != (size_t)tgt_len) {
        return false;
    }

    double acc = 0.0, comp = 0.0; int cnt = 0;
    for (int b = 0; b < batch; ++b) {
        const double* xi = in + (size_t)b * in_len;
        const double* ti = tgt + (size_t)b * tgt_len;
        double mse = 0.0;
        if (!n->train_one(xi, in_len, ti, tgt_len, lr, &mse)) {
            return false;
        }
        if (std::isfinite(mse)) {
            const double t = acc + mse;
            if (std::abs(acc) >= std::abs(mse)) {
                comp += (acc - t) + mse;
            }
            else {
                comp += (mse - t) + acc;
            }
            acc = t; ++cnt;
        }
    }
    if (mean_mse) {
        const double sum = acc + comp;
        *mean_mse = (cnt > 0 ? sum / (double)cnt : 0.0);
    }
    return true;
}

DLL_EXTERN bool  DLL_CALL NN_GetWeights(int h, int i, double* W, int Wlen, double* b, int blen) {
    NeuralNetwork* net = get_net(h); return net ? net->get_weights(i, W, Wlen, b, blen) : false;
}
DLL_EXTERN bool  DLL_CALL NN_SetWeights(int h, int i, const double* W, int Wlen, const double* b, int blen) {
    NeuralNetwork* net = get_net(h); return net ? net->set_weights(i, W, Wlen, b, blen) : false;
}

// ============================================================================
// Exported C API — Offline training on full series (MLP)
// ============================================================================
DLL_EXTERN void DLL_CALL NN_TrainSeries(
    int h,
    const double* X, int x_len,          // vstupní řada (1D, normalizovaná)
    const double* Y, int y_len,          // cílová řada (1D, normalizovaná)
    int win_in, int win_out, int lead,   // okno vstupu, délka výstupu, posun predikce
    int epochs, double lr, double target_mse,
    int batch_size, int shuffle          // 0/1
) {
    try {
        nnctl::clear_err(nnctl::g_err_nn, h);
        NeuralNetwork* n = get_net(h);
        if (!n) {
            nnctl::set_err(nnctl::g_err_nn, h, 1, L"NN handle not found"); return;
        }

        if (!X || !Y || x_len <= 0 || y_len <= 0) {
            nnctl::set_err(nnctl::g_err_nn, h, 2, L"Null data or non-positive length"); return;
        }
        if (win_in <= 0 || win_out <= 0 || lead < 0) {
            nnctl::set_err(nnctl::g_err_nn, h, 3, L"Invalid window/lead"); return;
        }
        if ((int)n->in_size() != win_in || (int)n->out_size() != win_out) {
            nnctl::set_err(nnctl::g_err_nn, h, 4, L"Topology mismatch: in/out size vs win_in/win_out");
            return;
        }
        const int term_sum = win_in + lead + win_out;
        const int max_t_x = x_len - term_sum + 1;
        const int max_t_y = y_len - (lead + win_out);
        const int N = static_cast<int>(std::min(max_t_x, max_t_y));

        if (N <= 0) {
            nnctl::set_err(nnctl::g_err_nn, h, 5, L"Series too short for given windows"); return;
        }

        if (epochs <= 0) epochs = 1;
        if (batch_size <= 0) batch_size = 32;
        if (lr <= 0.0) lr = 1e-3;
        if (!std::isfinite(target_mse) || target_mse < 0.0) target_mse = 0.0;

        // indexy vzorků
        std::vector<int> idx(N);
        for (int i = 0; i < N; ++i) idx[i] = i;

        std::vector<double> bin, btgt;
        bin.resize((size_t)batch_size * (size_t)win_in);
        btgt.resize((size_t)batch_size * (size_t)win_out);

        for (int e = 0; e < epochs; ++e) { // Hlavní smyčka epoch
            if (shuffle) {
                rng::shuffle(idx.begin(), idx.end());
            }

            double acc = 0.0, comp = 0.0; int cnt = 0;

            for (int p = 0; p < N; p += batch_size) { // Smyčka přes dávky
                if (nnctl::g_cancel_nn[h].load() == 1) {
                    return;
                }
                const int bs = std::min(batch_size, N - p);

                // naplnit batch
                for (int b = 0; b < bs; ++b) {
                    const int t0 = idx[p + b];
                    const double* x0 = X + (size_t)t0;
                    const double* y0 = Y + (size_t)(t0 + win_in + lead);
                    std::memcpy(&bin[(size_t)b * win_in], x0, (size_t)win_in * sizeof(double));
                    std::memcpy(&btgt[(size_t)b * win_out], y0, (size_t)win_out * sizeof(double));
                }

                double mean_mse = 0.0;
                if (!NN_TrainBatch(h, bin.data(), bs, win_in, btgt.data(), win_out, lr, &mean_mse)) {
                    nnctl::set_err(nnctl::g_err_nn, h, 6, L"NN_TrainBatch failed");
                    return;
                }

                // kumulace MSE (kompenzované sčítání)
                const double t = acc + mean_mse;
                if (std::abs(acc) >= std::abs(mean_mse)) {
                    comp += (acc - t) + mean_mse;
                }
                else {
                    comp += (mean_mse - t) + acc;
                }
                acc = t; ++cnt;

                // MSE monitor (lehce řidší frekvence)
                if ((p / batch_size) % 8 == 0) {
                    NN_MSE_Push(mean_mse);
                }
            }

            const double epoch_mse = (cnt > 0 ? acc + comp : 0.0) / (double)std::max(1, cnt);
            NN_MSE_Push(epoch_mse);
            if (epoch_mse <= target_mse && target_mse > 0.0) {
                return;
            }
        }
    }
    catch (const std::exception& ex) {
        std::wstring w = utf8_to_wide(ex.what());
        nnctl::set_err(nnctl::g_err_nn, h, 100, w.c_str());
    }
    catch (...) {
        nnctl::set_err(nnctl::g_err_nn, h, 101, L"Unknown exception in NN_TrainSeries");
    }
}


// ============================================================================
// Exported C API — LSTM Inference
// ============================================================================
DLL_EXTERN int DLL_CALL LSTM_Create(int in_sz, int hid_sz, int out_sz, int num_layers) {
    if (in_sz <= 0 || hid_sz <= 0 || out_sz <= 0 || num_layers <= 0) {
        return 0;
    }
    return lstm::lstm_alloc((size_t)in_sz, (size_t)hid_sz, (size_t)out_sz, (size_t)num_layers);
}

DLL_EXTERN void DLL_CALL LSTM_Free(int h) { lstm::lstm_free(h); }

DLL_EXTERN int DLL_CALL LSTM_Reset(int h) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return 0;
    }
    n->reset_state();
    return 1;
}

DLL_EXTERN int DLL_CALL LSTM_GetInfo(int h, int* in_sz, int* hid_sz, int* out_sz, int* num_layers) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return 0;
    }
    if (in_sz)      *in_sz = (int)n->in_sz;
    if (hid_sz)     *hid_sz = (int)n->hid_sz;
    if (out_sz)     *out_sz = (int)n->out_sz;
    if (num_layers) *num_layers = (int)n->layers_n;
    return 1;
}

DLL_EXTERN bool DLL_CALL LSTM_ForwardLast(int h, const double* seq, int seq_len, double* out, int out_len) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    return n->forward_last(seq, seq_len, out, out_len);
}

DLL_EXTERN bool DLL_CALL LSTM_ForwardSeq(int h, const double* seq, int seq_len, double* out, int out_len) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    return n->forward_seq(seq, seq_len, out, out_len);
}

// ============================================================================
// Exported C API — Offline training on full series (LSTM)
// ============================================================================
DLL_EXTERN void DLL_CALL LSTM_TrainSeries(
    int h,
    const double* X, int x_len,
    const double* Y, int y_len,
    int win_in, int win_out, int lead,
    int epochs, double lr, double target_mse,
    int batch_size, int tbptt_k, int shuffle
) {
    try {
        nnctl::clear_err(nnctl::g_err_lstm, h);
        lstm::LSTMNet* n = lstm::lstm_get(h);
        if (!n) {
            nnctl::set_err(nnctl::g_err_lstm, h, 1, L"LSTM handle not found"); return;
        }

        if (!X || !Y || x_len <= 0 || y_len <= 0) {
            nnctl::set_err(nnctl::g_err_lstm, h, 2, L"Null data or non-positive length"); return;
        }
        if (win_in <= 0 || win_out <= 0 || lead < 0) {
            nnctl::set_err(nnctl::g_err_lstm, h, 3, L"Invalid window/lead"); return;
        }

        const int actual_in_sz = (int)n->in_sz;
        if (actual_in_sz <= 0) {
            nnctl::set_err(nnctl::g_err_lstm, h, 4, L"Invalid LSTM in_sz"); return;
        }
        if ((int)n->out_sz != win_out) {
            nnctl::set_err(nnctl::g_err_lstm, h, 5, L"Topology mismatch: out_sz vs win_out"); return;
        }

        const int term_sum_lstm = win_in + lead + win_out;
        const int max_t_x = x_len - term_sum_lstm + 1;
        const int max_t_y = y_len - (lead + win_out);
        const int N = static_cast<int>(std::min(max_t_x, max_t_y));

        if (N <= 0) {
            nnctl::set_err(nnctl::g_err_lstm, h, 6, L"Series too short for given windows"); return;
        }

        if (epochs <= 0) epochs = 1;
        if (batch_size <= 0) batch_size = 16; // LSTM mívá menší batch
        if (lr <= 0.0) lr = 1e-3;
        if (!std::isfinite(target_mse) || target_mse < 0.0) target_mse = 0.0;

        // TBPTT nastavení
        LSTM_SetTBPTT(h, tbptt_k);

        // indexy vzorků
        std::vector<int> idx(N);
        for (int i = 0; i < N; ++i) idx[i] = i;

        // buffery: seq_batch má tvar [batch, win_in, actual_in_sz]
        std::vector<double> seq_batch((size_t)batch_size * (size_t)win_in * (size_t)actual_in_sz);
        std::vector<double> tgt_batch((size_t)batch_size * (size_t)win_out);

        for (int e = 0; e < epochs; ++e) { // Hlavní smyčka epoch
            if (shuffle) {
                rng::shuffle(idx.begin(), idx.end());
            }

            double acc = 0.0, comp = 0.0; int cnt = 0;

            for (int p = 0; p < N; p += batch_size) { // Smyčka přes dávky
                if (nnctl::g_cancel_lstm[h].load() == 1) {
                    return;
                }
                const int bs = std::min(batch_size, N - p);

                // naplnit batch
                for (int b = 0; b < bs; ++b) {
                    const int t0 = idx[p + b];
                    if (actual_in_sz == 1) {
                        const double* x0 = X + (size_t)t0;
                        double* dst = &seq_batch[(size_t)b * (size_t)win_in * (size_t)actual_in_sz];
                        for (int t = 0; t < win_in; ++t) {
                            dst[(size_t)t * (size_t)actual_in_sz + 0] = x0[t];
                        }
                    }
                    else {
                        const double* x0 = X + (size_t)t0 * (size_t)actual_in_sz;
                        double* dst = &seq_batch[(size_t)b * (size_t)win_in * (size_t)actual_in_sz];
                        std::memcpy(dst, x0, (size_t)win_in * (size_t)actual_in_sz * sizeof(double));
                    }

                    const double* y0 = Y + (size_t)(t0 + win_in + lead);
                    std::memcpy(&tgt_batch[(size_t)b * (size_t)win_out], y0, (size_t)win_out * sizeof(double));
                }

                double mean_mse = 0.0;
                if (!LSTM_TrainBatch(h, seq_batch.data(), bs, win_in, tgt_batch.data(), win_out, lr, &mean_mse)) {
                    nnctl::set_err(nnctl::g_err_lstm, h, 7, L"LSTM_TrainBatch failed");
                    return;
                }

                const double t = acc + mean_mse;
                if (std::abs(acc) >= std::abs(mean_mse)) {
                    comp += (acc - t) + mean_mse;
                }
                else {
                    comp += (mean_mse - t) + acc;
                }
                acc = t; ++cnt;

                if ((p / batch_size) % 8 == 0) {
                    NN_MSE_Push(mean_mse);
                }
            }

            const double epoch_mse = (cnt > 0 ? acc + comp : 0.0) / (double)std::max(1, cnt);
            NN_MSE_Push(epoch_mse);
            if (epoch_mse <= target_mse && target_mse > 0.0) {
                return;
            }
        }
    }
    catch (const std::exception& ex) {
        std::wstring w = utf8_to_wide(ex.what());
        nnctl::set_err(nnctl::g_err_lstm, h, 100, w.c_str());
    }
    catch (...) {
        nnctl::set_err(nnctl::g_err_lstm, h, 101, L"Unknown exception in LSTM_TrainSeries");
    }
}


// ============================================================================
// Exported C API (cdecl) — LSTM Weights (inference-time helpers)
// ============================================================================
DLL_EXTERN bool DLL_CALL LSTM_SetOutWeights(int h, const double* W, int Wlen, const double* b, int blen) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    const int needW = (int)(n->out_sz * n->hid_sz);
    const int needb = (int)n->out_sz;
    if (Wlen != needW || blen != needb) {
        return false;
    }
    std::memcpy(n->W_out.data(), W, (size_t)needW * sizeof(double));
    std::memcpy(n->b_out.data(), b, (size_t)needb * sizeof(double));
    return true;
}
DLL_EXTERN bool DLL_CALL LSTM_GetOutWeights(int h, double* W, int Wlen, double* b, int blen) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    const int needW = (int)(n->out_sz * n->hid_sz);
    const int needb = (int)n->out_sz;
    if (Wlen != needW || blen != needb) {
        return false;
    }
    std::memcpy(W, n->W_out.data(), (size_t)needW * sizeof(double));
    std::memcpy(b, n->b_out.data(), (size_t)needb * sizeof(double));
    return true;
}

DLL_EXTERN bool DLL_CALL LSTM_SetLayerWeights(int h, int layer_index,
    const double* Wx, int Wxlen, const double* Wh, int Whlen, const double* b, int blen)
{
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    if (layer_index < 0 || (size_t)layer_index >= n->layers_n) {
        return false;
    }
    lstm::LSTMLayer& L = n->layers[(size_t)layer_index];
    const int needWx = (int)(4 * L.hid_sz * L.in_sz);
    const int needWh = (int)(4 * L.hid_sz * L.hid_sz);
    const int needb = (int)(4 * L.hid_sz);
    if (Wxlen != needWx || Whlen != needWh || blen != needb) {
        return false;
    }
    std::memcpy(L.W_x.data(), Wx, (size_t)needWx * sizeof(double));
    std::memcpy(L.W_h.data(), Wh, (size_t)needWh * sizeof(double));
    std::memcpy(L.b.data(), b, (size_t)needb * sizeof(double));
    return true;
}

DLL_EXTERN bool DLL_CALL LSTM_GetLayerWeights(int h, int layer_index,
    double* Wx, int Wxlen, double* Wh, int Whlen, double* b, int blen)
{
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    if (layer_index < 0 || (size_t)layer_index >= n->layers_n) {
        return false;
    }
    lstm::LSTMLayer& L = n->layers[(size_t)layer_index];
    const int needWx = (int)(4 * L.hid_sz * L.in_sz);
    const int needWh = (int)(4 * L.hid_sz * L.hid_sz);
    const int needb = (int)(4 * L.hid_sz);
    if (Wxlen != needWx || Whlen != needWh || blen != needb) {
        return false;
    }
    std::memcpy(Wx, L.W_x.data(), (size_t)needWx * sizeof(double));
    std::memcpy(Wh, L.W_h.data(), (size_t)needWh * sizeof(double));
    std::memcpy(b, L.b.data(), (size_t)needb * sizeof(double));
    return true;
}

// ============================================================================
// Exported C API (cdecl) — LSTM Training (NEW)
// ============================================================================
DLL_EXTERN int   DLL_CALL LSTM_SetTBPTT(int h, int k_steps) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return 0;
    }
    n->tbptt_k = (k_steps < 0 ? 0 : k_steps);
    return 1;
}
DLL_EXTERN int   DLL_CALL LSTM_SetClipGrad(int h, double clip_norm) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return 0;
    }
    n->clip_norm = (clip_norm < 0.0 ? 0.0 : clip_norm);
    return 1;
}

DLL_EXTERN bool  DLL_CALL LSTM_TrainOne(int h, const double* seq, int seq_len,
    const double* tgt, int tgt_len,
    double lr, double* mse_out) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    const bool ok = n->train_one(seq, seq_len, tgt, tgt_len, lr, mse_out);
    if (ok && mse_out) {
        std::lock_guard<std::mutex> lk(ui::g.mtx);
        ui::g.data.push_back(*mse_out);
        while (ui::g.data.size() > ui::g.max_points) {
            ui::g.data.pop_front();
        }
        if (ui::g.hwnd) {
            InvalidateRect(ui::g.hwnd, NULL, FALSE);
        }
    }
    return ok;
}

DLL_EXTERN bool  DLL_CALL LSTM_TrainBatch(int h, const double* seq_batch, int batch, int seq_len,
    const double* tgt_batch, int tgt_len,
    double lr, double* mean_mse) {
    lstm::LSTMNet* n = lstm::lstm_get(h);
    if (!n) {
        return false;
    }
    const bool ok = n->train_batch(seq_batch, batch, seq_len, tgt_batch, tgt_len, lr, mean_mse);
    if (ok && mean_mse) {
        std::lock_guard<std::mutex> lk(ui::g.mtx);
        ui::g.data.push_back(*mean_mse);
        while (ui::g.data.size() > ui::g.max_points) {
            ui::g.data.pop_front();
        }
        if (ui::g.hwnd) {
            InvalidateRect(ui::g.hwnd, NULL, FALSE);
        }
    }
    return ok;
}

// ----------------------------------------------------------------------------
// Exported C API — MSE monitor controls (optional from MQL5)
// ----------------------------------------------------------------------------
DLL_EXTERN void  DLL_CALL NN_MSE_Show(int show) {
    if (show) {
        ui::ensure_thread();
        ui::g.visible = true;
        if (ui::g.hwnd) {
            ShowWindow(ui::g.hwnd, SW_SHOW);
            // Keep window on top when shown again
            SetWindowPos(ui::g.hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                SWP_NOMOVE | SWP_NOSIZE);
            SetForegroundWindow(ui::g.hwnd);
        }
    }
    else {
        ui::g.visible = false;
        if (ui::g.hwnd) {
            ShowWindow(ui::g.hwnd, SW_HIDE);
        }
    }
}

DLL_EXTERN void  DLL_CALL NN_MSE_Push(double mse) {
    std::lock_guard<std::mutex> lk(ui::g.mtx);
    ui::g.data.push_back(mse);
    while (ui::g.data.size() > ui::g.max_points) {
        ui::g.data.pop_front();
    }
    if (ui::g.hwnd) {
        InvalidateRect(ui::g.hwnd, NULL, FALSE);
    }
}

DLL_EXTERN void  DLL_CALL NN_MSE_Clear() {
    std::lock_guard<std::mutex> lk(ui::g.mtx);
    ui::g.data.clear();
    if (ui::g.hwnd) {
        InvalidateRect(ui::g.hwnd, NULL, TRUE);
    }
}

DLL_EXTERN void  DLL_CALL NN_MSE_SetMaxPoints(int n) {
    if (n < 10) {
        n = 10;
    }
    std::lock_guard<std::mutex> lk(ui::g.mtx);
    ui::g.max_points = (size_t)n;
    while (ui::g.data.size() > ui::g.max_points) {
        ui::g.data.pop_front();
    }
    if (ui::g.hwnd) {
        InvalidateRect(ui::g.hwnd, NULL, TRUE);
    }
}

DLL_EXTERN void  DLL_CALL NN_MSE_SetAutoScale(int enable, double y_min, double y_max) {
    std::lock_guard<std::mutex> lk(ui::g.mtx);
    ui::g.autoscale = (enable != 0);
    if (!ui::g.autoscale) {
        ui::g.y_min = y_min; ui::g.y_max = y_max;
    }
    if (ui::g.hwnd) {
        InvalidateRect(ui::g.hwnd, NULL, TRUE);
    }
}

// ============================================================================
// Exported C API — Error / Cancel helpers
// ============================================================================
DLL_EXTERN int DLL_CALL NN_GetLastError(int h, int* code, wchar_t* buf, int buf_len) {
    std::lock_guard<std::mutex> lk(nnctl::g_err_mtx);
    auto it = nnctl::g_err_nn.find(h);
    if (it == nnctl::g_err_nn.end()) {
        return 0;
    }
    if (code) {
        *code = it->second.code;
    }
    if (buf && buf_len > 0) {
        int n = (int)std::min<size_t>(it->second.msg.size(), (size_t)buf_len - 1);
        wcsncpy_s(buf, buf_len, it->second.msg.c_str(), n);
    }
    return 1;
}
DLL_EXTERN void DLL_CALL NN_SetCancel(int h, int cancel_flag) {
    nnctl::g_cancel_nn[h].store(cancel_flag ? 1 : 0);
}
DLL_EXTERN int DLL_CALL LSTM_GetLastError(int h, int* code, wchar_t* buf, int buf_len) {
    std::lock_guard<std::mutex> lk(nnctl::g_err_mtx);
    auto it = nnctl::g_err_lstm.find(h);
    if (it == nnctl::g_err_lstm.end()) {
        return 0;
    }
    if (code) {
        *code = it->second.code;
    }
    if (buf && buf_len > 0) {
        int n = (int)std::min<size_t>(it->second.msg.size(), (size_t)buf_len - 1);
        wcsncpy_s(buf, buf_len, it->second.msg.c_str(), n);
    }
    return 1;
}
DLL_EXTERN void DLL_CALL LSTM_SetCancel(int h, int cancel_flag) {
    nnctl::g_cancel_lstm[h].store(cancel_flag ? 1 : 0);
}
DLL_EXTERN void DLL_CALL NN_SetSeed(unsigned int s) {
    std::srand(s);
    rng::seed(s);
}


// ----------------------------------------------------------------------------
// DllMain
// ----------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID) {
    if (reason == DLL_PROCESS_ATTACH) {
        g_hInst = (HINSTANCE)hModule;
        DisableThreadLibraryCalls(hModule);
        // std::srand(1234567); // uncomment for deterministic init
    }
    else if (reason == DLL_PROCESS_DETACH) {
        ui::shutdown();
    }
    return TRUE;
}
