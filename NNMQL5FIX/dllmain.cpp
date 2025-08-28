// ============================================================================
//  dllmain.cpp — Lehká MLP knihovna jako DLL pro MQL5 (x64, MSVC)
//  --------------------------------------------------------------------------
//  Max-precision build: stabilní sigmoid, přesné dot produkty (Neumaier + FMA),
//  kompenzované sumace MSE a batch MSE, explicitní konverze typů.
//  API beze změny. Paralelismus vypnutý (sekvenční).
//  (c) 2025, MIT-like spirit — použijte, ale prosíme: uvádějte autorství.
// ============================================================================

// *** MUSÍ být první řádek kvůli PCH ***
#include "pch.h"

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

#define DLL_EXPORT extern "C" __declspec(dllexport)

// ----------------------------------------------------------------------------
// Pomocné numerické utility (maximální přesnost)
// ----------------------------------------------------------------------------
namespace precise {
    inline double sigmoid(double x) {
        if (x >= 0.0) {
            double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        else {
            double z = std::exp(x);
            return z / (1.0 + z);
        }
    }

    inline double neumaier_sum_accumulate(double sum, double add, double& comp) {
        double t = sum + add;
        if (std::abs(sum) >= std::abs(add)) comp += (sum - t) + add;
        else                                comp += (add - t) + sum;
        return t;
    }

    inline double dot_neumaier_fma(const double* a, const double* b, size_t n) {
        double sum = 0.0, c = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double prod = std::fma(a[i], b[i], 0.0);
            sum = neumaier_sum_accumulate(sum, prod, c);
        }
        return sum + c;
    }

    inline double sum_of_squares(const std::vector<double>& v) {
        double sum = 0.0, c = 0.0;
        for (double x : v) {
            double sq = std::fma(x, x, 0.0);
            sum = neumaier_sum_accumulate(sum, sq, c);
        }
        return sum + c;
    }
}

// ----------------------------------------------------------------------------
// Tensor & Matrix
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
// Aktivace
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
    std::vector<double> last_in, last_z, last_out;

    DenseLayer(size_t inSize, size_t outSize, ActKind k)
        : in_sz(inSize), out_sz(outSize), W(outSize, inSize), b(outSize, 0.0), act(k) {
        const double denom = (double)std::max<size_t>(1, in_sz);
        const double scale = (k == ActKind::RELU) ? std::sqrt(2.0 / denom) : std::sqrt(1.0 / denom);
        for (double& w : W.a) {
            double u = (std::rand() / (double)RAND_MAX) * 2.0 - 1.0;
            w = u * scale;
        }
    }

    std::vector<double> forward(const std::vector<double>& x) {
        if (x.size() != in_sz) throw std::runtime_error("Dense forward: bad input size");
        last_in = x;
        last_z.assign(out_sz, 0.0);
        last_out.assign(out_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            const double* wrow = &W.a[o * in_sz];
            double dot = precise::dot_neumaier_fma(wrow, x.data(), in_sz);
            double z = b[o] + dot;
            last_z[o] = z;
            last_out[o] = Activation::f(act, z);
        }
        return last_out;
    }

    std::vector<double> backward(const std::vector<double>& dL_dy, double lr) {
        std::vector<double> dL_dz(out_sz);
        for (size_t o = 0; o < out_sz; ++o) {
            double y = last_out[o];
            double z = last_z[o];
            dL_dz[o] = dL_dy[o] * Activation::df(act, y, z);
        }
        const double gclip = 5.0;
        for (double& g : dL_dz) { if (g > gclip)g = gclip; if (g < -gclip)g = -gclip; }
        for (size_t o = 0; o < out_sz; ++o) {
            b[o] -= lr * dL_dz[o];
            double* wrow = &W.a[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) {
                double prod = std::fma(dL_dz[o], last_in[i], 0.0);
                wrow[i] = std::fma(-lr, prod, wrow[i]);
            }
        }
        std::vector<double> dL_dx(in_sz, 0.0), comp(in_sz, 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            const double* wrow = &W.a[o * in_sz];
            double go = dL_dz[o];
            for (size_t i = 0; i < in_sz; ++i) {
                double add = std::fma(wrow[i], go, 0.0);
                double t = dL_dx[i] + add;
                if (std::abs(dL_dx[i]) >= std::abs(add))comp[i] += (dL_dx[i] - t) + add;
                else comp[i] += (add - t) + dL_dx[i];
                dL_dx[i] = t;
            }
        }
        for (size_t i = 0; i < in_sz; ++i) dL_dx[i] += comp[i];
        return dL_dx;
    }
};

// ----------------------------------------------------------------------------
// MSELoss
// ----------------------------------------------------------------------------
struct MSELoss {
    static double loss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) throw std::runtime_error("MSE size mismatch");
        std::vector<double> e(y.size());
        for (size_t i = 0; i < y.size(); ++i) e[i] = y[i] - t[i];
        double ss = precise::sum_of_squares(e);
        return ss / (double)y.size();
    }
    static std::vector<double> dloss(const std::vector<double>& y, const std::vector<double>& t) {
        if (y.size() != t.size()) throw std::runtime_error("MSE size mismatch");
        std::vector<double> g(y.size());
        double n = (double)y.size();
        for (size_t i = 0; i < y.size(); ++i)
            g[i] = std::fma(2.0, (y[i] - t[i]), 0.0) / n;
        return g;
    }
};

// ----------------------------------------------------------------------------
// NeuralNetwork
// ----------------------------------------------------------------------------
class NeuralNetwork {
    std::vector<std::unique_ptr<DenseLayer>> layers;
    size_t input_size{ 0 }, output_size{ 0 };
public:
    bool add_dense(size_t in_sz, size_t out_sz, ActKind k) {
        if (layers.empty()) input_size = in_sz;
        else if (layers.back()->out_sz != in_sz) return false;
        layers.emplace_back(std::make_unique<DenseLayer>(in_sz, out_sz, k));
        output_size = out_sz;
        return true;
    }
    size_t in_size() const { return input_size; }
    size_t out_size()const { return output_size; }
    bool forward(const double* in, int in_len, double* out, int out_len) {
        if ((int)input_size != in_len || (int)output_size != out_len || layers.empty())return false;
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers) x = L->forward(x);
        for (int i = 0; i < out_len; ++i) out[i] = x[(size_t)i];
        return true;
    }
    bool train_one(const double* in, int in_len, const double* tgt, int tgt_len, double lr, double* mse = nullptr) {
        if ((int)input_size != in_len || (int)output_size != tgt_len || layers.empty())return false;
        std::vector<double> x(in, in + in_len);
        for (auto& L : layers)x = L->forward(x);
        std::vector<double> t(tgt, tgt + tgt_len);
        if (mse) *mse = MSELoss::loss(x, t);
        std::vector<double> g = MSELoss::dloss(x, t);
        for (int li = (int)layers.size() - 1; li >= 0; --li) g = layers[(size_t)li]->backward(g, lr);
        return true;
    }
    bool get_weights(int i, double* W, int Wlen, double* b, int blen)const {
        if (i < 0)return false; size_t u = (size_t)i;
        if (u >= layers.size())return false;
        const DenseLayer* L = layers[u].get(); if (!L)return false;
        size_t needW = L->out_sz * L->in_sz; size_t needb = L->out_sz;
        if ((size_t)Wlen != needW || (size_t)blen != needb)return false;
        std::memcpy(W, L->W.a.data(), needW * sizeof(double));
        std::memcpy(b, L->b.data(), needb * sizeof(double));
        return true;
    }
    bool set_weights(int i, const double* W, int Wlen, const double* b, int blen) {
        if (i < 0)return false; size_t u = (size_t)i;
        if (u >= layers.size())return false;
        DenseLayer* L = layers[u].get(); if (!L)return false;
        size_t needW = L->out_sz * L->in_sz; size_t needb = L->out_sz;
        if ((size_t)Wlen != needW || (size_t)blen != needb)return false;
        std::memcpy(L->W.a.data(), W, needW * sizeof(double));
        std::memcpy(L->b.data(), b, needb * sizeof(double));
        return true;
    }
};

// ----------------------------------------------------------------------------
// Instance management
// ----------------------------------------------------------------------------
static std::unordered_map<int, std::unique_ptr<NeuralNetwork>> g_nets;
static std::mutex g_mtx;
static int g_next_handle = 1;
static int alloc_handle() { std::lock_guard<std::mutex>lk(g_mtx); int h = g_next_handle++; g_nets.emplace(h, std::make_unique<NeuralNetwork>()); return h; }
static NeuralNetwork* get_net(int h) { std::lock_guard<std::mutex>lk(g_mtx); auto it = g_nets.find(h); return it == g_nets.end() ? nullptr : it->second.get(); }
static void free_handle(int h) { std::lock_guard<std::mutex>lk(g_mtx); g_nets.erase(h); }

// ----------------------------------------------------------------------------
// Exportované API
// ----------------------------------------------------------------------------
DLL_EXPORT int NN_Create() { return alloc_handle(); }
DLL_EXPORT void NN_Free(int h) { free_handle(h); }
DLL_EXPORT bool NN_AddDense(int h, int inSz, int outSz, int act) {
    NeuralNetwork* net = get_net(h); if (!net)return false;
    ActKind k = (act == 0 ? ActKind::SIGMOID : act == 1 ? ActKind::RELU : act == 2 ? ActKind::TANH : act == 4 ? ActKind::SYM_SIG : ActKind::LINEAR);
    return net->add_dense((size_t)inSz, (size_t)outSz, k);
}
DLL_EXPORT int NN_InputSize(int h) { auto* n = get_net(h); return n ? (int)n->in_size() : 0; }
DLL_EXPORT int NN_OutputSize(int h) { auto* n = get_net(h); return n ? (int)n->out_size() : 0; }
DLL_EXPORT bool NN_Forward(int h, const double* in, int in_len, double* out, int out_len) { auto* n = get_net(h); return n ? n->forward(in, in_len, out, out_len) : false; }
DLL_EXPORT bool NN_TrainOne(int h, const double* in, int in_len, const double* tgt, int tgt_len, double lr, double* mse) { auto* n = get_net(h); return n ? n->train_one(in, in_len, tgt, tgt_len, lr, mse) : false; }
DLL_EXPORT bool NN_ForwardBatch(int h, const double* in, int batch, int in_len, double* out, int out_len) {
    if (batch <= 0)return false; NeuralNetwork* n = get_net(h); if (!n)return false;
    if (n->in_size() != (size_t)in_len || n->out_size() != (size_t)out_len)return false;
    for (int b = 0; b < batch; ++b) { const double* xi = in + (size_t)b * in_len; double* yi = out + (size_t)b * out_len; if (!n->forward(xi, in_len, yi, out_len))return false; }
    return true;
}
DLL_EXPORT bool NN_TrainBatch(int h, const double* in, int batch, int in_len, const double* tgt, int tgt_len, double lr, double* mean_mse) {
    if (batch <= 0)return false; NeuralNetwork* n = get_net(h); if (!n)return false;
    if (n->in_size() != (size_t)in_len || n->out_size() != (size_t)tgt_len)return false;
    double acc = 0.0, comp = 0.0; int cnt = 0;
    for (int b = 0; b < batch; ++b) {
        const double* xi = in + (size_t)b * in_len; const double* ti = tgt + (size_t)b * tgt_len;
        double mse = 0.0; if (!n->train_one(xi, in_len, ti, tgt_len, lr, &mse))return false;
        if (std::isfinite(mse)) { double t = acc + mse; if (std::abs(acc) >= std::abs(mse))comp += (acc - t) + mse; else comp += (mse - t) + acc; acc = t; ++cnt; }
    }
    if (mean_mse) { double sum = acc + comp; *mean_mse = (cnt > 0 ? sum / (double)cnt : 0.0); }
    return true;
}
DLL_EXPORT bool NN_GetWeights(int h, int i, double* W, int Wlen, double* b, int blen) { NeuralNetwork* net = get_net(h); return net ? net->get_weights(i, W, Wlen, b, blen) : false; }
DLL_EXPORT bool NN_SetWeights(int h, int i, const double* W, int Wlen, const double* b, int blen) { NeuralNetwork* net = get_net(h); return net ? net->set_weights(i, W, Wlen, b, blen) : false; }

// ----------------------------------------------------------------------------
// DllMain
// ----------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE, DWORD, LPVOID) { return TRUE; }
