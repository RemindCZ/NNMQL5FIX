// ============================================================================
//  dllmain.cpp — High-Performance MLP DLL for MQL5 (x64, MSVC)
//  --------------------------------------------------------------------------
//  VERSION: 4.0 (Robust ABI, Zero-Alloc Training Loop, Safe Load/Save, OMP reduce)
//  NOTES:
//   - Exportní API zachováno (NN_Create, NN_AddDense, NN_TrainBatch, ...)
//   - Calling convention sjednocena na __stdcall (pro MQL5 nejbezpečnější).
//   - Trénink: žádné alokace uvnitř batch/layer smyček.
//   - MSE: průměr přes batch i output_size (skutečné "mean squared error").
//   - Load: nikdy nevrací true při částečném načtení.
//   - UI: robustnější ošetření selhání CreateWindow/Timer.
// ============================================================================

#define NOMINMAX

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
#include <cstdio>

#include <omp.h> // /openmp

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------
#ifndef DLL_EXTERN
#define DLL_EXTERN extern "C" __declspec(dllexport)
#endif

#ifndef DLL_CALL
#define DLL_CALL __stdcall
#endif

// Bezpečná bariéra proti výjimkám (MetaTrader nesnáší C++ výjimky přes ABI)
#define DLL_CATCH_ALL(retval) \
    catch (const std::exception&) { return retval; } \
    catch (...) { return retval; }

// ---------------------------------------------------------------------------
// Hyperparametry
// ---------------------------------------------------------------------------
struct HyperParams {
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.01;
};

static HINSTANCE g_hInst = nullptr;

// ---------------------------------------------------------------------------
// RNG (rychlé + thread-safe bez globálního mutexu v horké smyčce)
// ---------------------------------------------------------------------------
namespace rng {
    static std::atomic<uint64_t> g_seed{ 1234567ULL };

    inline void seed(uint64_t s) {
        g_seed.store(s ? s : 1ULL, std::memory_order_relaxed);
    }

    inline uint64_t base_seed() {
        return g_seed.load(std::memory_order_relaxed);
    }

    inline double uniform(double minv, double maxv) {
        thread_local std::mt19937_64 gen{ base_seed() ^ (uint64_t)(uintptr_t)&gen };
        std::uniform_real_distribution<double> dist(minv, maxv);
        return dist(gen);
    }
}

// ---------------------------------------------------------------------------
// Precizní matematika
// ---------------------------------------------------------------------------
namespace precise {
    inline double sigmoid(double x) {
        // numericky stabilní sigmoid
        if (x >= 0.0) {
            const double z = std::exp(-x);
            return 1.0 / (1.0 + z);
        }
        else {
            const double z = std::exp(x);
            return z / (1.0 + z);
        }
    }

    // Neumaier summation (lepší než Kahan pro různá měřítka)
    inline double neumaier_add(double sum, double val, double& comp) {
        const double t = sum + val;
        if (std::abs(sum) >= std::abs(val)) comp += (sum - t) + val;
        else                                 comp += (val - t) + sum;
        return t;
    }

    inline double dot_product(const double* a, const double* b, size_t n) {
        double sum = 0.0, c = 0.0;
        for (size_t i = 0; i < n; ++i) {
            const double prod = a[i] * b[i];
            sum = neumaier_add(sum, prod, c);
        }
        return sum + c;
    }
}

// ---------------------------------------------------------------------------
// AdamW (stav momentů)
// ---------------------------------------------------------------------------
struct AdamState {
    std::vector<double> m;
    std::vector<double> v;

    void resize(size_t n) {
        m.assign(n, 0.0);
        v.assign(n, 0.0);
    }
    void reset() {
        std::fill(m.begin(), m.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
    }

    void update(std::vector<double>& params,
        const std::vector<double>& grads,
        const HyperParams& hp,
        uint64_t t)
    {
        const size_t n = params.size();
        if (grads.size() != n || m.size() != n || v.size() != n) return;

        const double bc1 = 1.0 - std::pow(hp.beta1, (double)t);
        const double bc2 = 1.0 - std::pow(hp.beta2, (double)t);
        const double inv_bc1 = (bc1 != 0.0) ? (1.0 / bc1) : 1.0;
        const double inv_bc2 = (bc2 != 0.0) ? (1.0 / bc2) : 1.0;

        for (size_t i = 0; i < n; ++i) {
            const double g = grads[i];

            // Decoupled weight decay
            params[i] -= hp.lr * hp.weight_decay * params[i];

            m[i] = hp.beta1 * m[i] + (1.0 - hp.beta1) * g;
            v[i] = hp.beta2 * v[i] + (1.0 - hp.beta2) * (g * g);

            const double m_hat = m[i] * inv_bc1;
            const double v_hat = v[i] * inv_bc2;

            params[i] -= hp.lr * m_hat / (std::sqrt(v_hat) + hp.eps);
        }
    }
};

// ---------------------------------------------------------------------------
// Aktivace
// ---------------------------------------------------------------------------
enum class ActKind : int { SIGMOID = 0, RELU = 1, TANH = 2, LINEAR = 3, SYM_SIG = 4 };

struct Activation {
    static double f(ActKind k, double x) {
        switch (k) {
        case ActKind::SIGMOID: return precise::sigmoid(x);
        case ActKind::RELU:    return (x > 0.0) ? x : 0.0;
        case ActKind::TANH:    return std::tanh(x);
        case ActKind::SYM_SIG: return 2.0 * precise::sigmoid(x) - 1.0;
        default:               return x; // LINEAR
        }
    }
    // derivace d(a)/d(z) — pro ReLU používáme z (nebo x), u ostatních stačí a
    static double df(ActKind k, double a, double z) {
        switch (k) {
        case ActKind::SIGMOID: return a * (1.0 - a);
        case ActKind::RELU:    return (z > 0.0) ? 1.0 : 0.0;
        case ActKind::TANH:    return 1.0 - a * a;
        case ActKind::SYM_SIG: return 0.5 * (1.0 - a * a);
        default:               return 1.0; // LINEAR
        }
    }
};

// ---------------------------------------------------------------------------
// Dense vrstva
// ---------------------------------------------------------------------------
struct DenseLayer {
    size_t in_sz = 0, out_sz = 0;
    std::vector<double> W; // [out][in] row-major
    std::vector<double> b; // [out]
    ActKind act = ActKind::LINEAR;
    AdamState adam_W, adam_b;

    DenseLayer(size_t in, size_t out, ActKind k) : in_sz(in), out_sz(out), act(k) {
        W.resize(in_sz * out_sz);
        b.assign(out_sz, 0.0);
        adam_W.resize(W.size());
        adam_b.resize(b.size());

        // He pro ReLU, Xavier pro ostatní (jednoduchá, ale rozumná volba)
        const double scale = (k == ActKind::RELU)
            ? std::sqrt(2.0 / (double)in_sz)
            : std::sqrt(1.0 / (double)in_sz);

        for (double& w : W) w = rng::uniform(-1.0, 1.0) * scale;
    }

    void forward(const double* x, double* z, double* a) const {
        for (size_t o = 0; o < out_sz; ++o) {
            double v = precise::dot_product(&W[o * in_sz], x, in_sz) + b[o];
            z[o] = v;
            a[o] = Activation::f(act, v);
        }
    }

    // Backward:
    //  dL/da (out_sz) + (x,in_sz,z,out_sz,a,out_sz) -> dL/dx (in_sz) + gradW + gradb
    void backward(const double* dL_da,
        const double* x,
        const double* z,
        const double* a,
        double* dL_dx,
        double* gradW,
        double* gradb) const
    {
        // dL/dz pro každý output neuron
        // zároveň bias grad
        // Pozn.: dL_dx musíme nulovat před akumulací
        std::fill(dL_dx, dL_dx + in_sz, 0.0);

        for (size_t o = 0; o < out_sz; ++o) {
            const double dz = dL_da[o] * Activation::df(act, a[o], z[o]);
            gradb[o] += dz;

            // gradW řádek o
            double* gw = &gradW[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) {
                gw[i] += dz * x[i];
            }

            // backprop do dL_dx
            const double* wrow = &W[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) {
                dL_dx[i] += wrow[i] * dz;
            }
        }
    }

    void update(const std::vector<double>& gradW,
        const std::vector<double>& gradb,
        const HyperParams& hp,
        uint64_t t)
    {
        adam_W.update(W, gradW, hp, t);
        adam_b.update(b, gradb, hp, t);
    }
};

// ---------------------------------------------------------------------------
// NeuralNetwork (MLP) — zero-alloc training loop
// ---------------------------------------------------------------------------
class NeuralNetwork {
    std::vector<std::unique_ptr<DenseLayer>> layers;
    size_t input_size = 0, output_size = 0;
    uint64_t time_step = 0;

    struct ThreadContext {
        // Buffery pro forward
        std::vector<std::vector<double>> Z; // [layer][out]
        std::vector<std::vector<double>> A; // [layer][out]
        std::vector<double> X0;             // vstup (kopie pro pohodlí)

        // Ping-pong delty (max(in/out) per layer, ale držíme zvlášť po layer)
        std::vector<double> delta_curr; // dim = out_size aktuální vrstvy (nebo output)
        std::vector<double> delta_prev; // dim = in_size aktuální vrstvy

        void resize(const std::vector<std::unique_ptr<DenseLayer>>& Ls, size_t in_sz) {
            Z.resize(Ls.size());
            A.resize(Ls.size());
            for (size_t i = 0; i < Ls.size(); ++i) {
                Z[i].assign(Ls[i]->out_sz, 0.0);
                A[i].assign(Ls[i]->out_sz, 0.0);
            }
            X0.assign(in_sz, 0.0);

            // delta buffery budou resize podle aktuální vrstvy při backprop (bez alokace ve smyčce)
            delta_curr.clear();
            delta_prev.clear();
        }

        void ensure_delta_sizes(size_t curr, size_t prev) {
            if (delta_curr.size() != curr) delta_curr.assign(curr, 0.0);
            else std::fill(delta_curr.begin(), delta_curr.end(), 0.0);

            if (delta_prev.size() != prev) delta_prev.assign(prev, 0.0);
            else std::fill(delta_prev.begin(), delta_prev.end(), 0.0);
        }
    };

    struct LayerGrads {
        std::vector<double> gW;
        std::vector<double> gb;

        void resize(size_t w_sz, size_t b_sz) {
            gW.assign(w_sz, 0.0);
            gb.assign(b_sz, 0.0);
        }
        void zero() {
            std::fill(gW.begin(), gW.end(), 0.0);
            std::fill(gb.begin(), gb.end(), 0.0);
        }
        void add_inplace(const LayerGrads& other) {
            const size_t nW = gW.size();
            const size_t nb = gb.size();
            for (size_t i = 0; i < nW; ++i) gW[i] += other.gW[i];
            for (size_t i = 0; i < nb; ++i) gb[i] += other.gb[i];
        }
    };

public:
    bool add_dense(size_t in_sz, size_t out_sz, ActKind k) {
        if (in_sz == 0 || out_sz == 0) return false;

        if (layers.empty()) input_size = in_sz;
        else if (layers.back()->out_sz != in_sz) return false;

        layers.emplace_back(std::make_unique<DenseLayer>(in_sz, out_sz, k));
        output_size = out_sz;
        return true;
    }

    size_t in_size() const { return input_size; }
    size_t out_size() const { return output_size; }

    bool forward(const double* in, int in_len, double* out, int out_len) {
        if (!in || !out) return false;
        if (layers.empty()) return false;
        if ((int)input_size != in_len || (int)output_size != out_len) return false;

        // Buffery bez zbytečných alokací ve smyčce: držíme dvě vektory a swap
        std::vector<double> x(input_size);
        std::memcpy(x.data(), in, sizeof(double) * input_size);

        std::vector<double> z, a;
        for (const auto& L : layers) {
            z.assign(L->out_sz, 0.0);
            a.assign(L->out_sz, 0.0);
            L->forward(x.data(), z.data(), a.data());
            x.swap(a);
        }

        std::memcpy(out, x.data(), sizeof(double) * output_size);
        return true;
    }

    bool train_batch(const double* in, int batch, int in_len,
        const double* tgt, int tgt_len,
        double lr, double* mean_mse)
    {
        if (!in || !tgt) return false;
        if (layers.empty()) return false;
        if (batch <= 0) return false;
        if (in_len != (int)input_size) return false;
        if (tgt_len != (int)output_size) return false;
        if (!(lr > 0.0) || !std::isfinite(lr)) return false;

        HyperParams hp;
        hp.lr = lr;

        ++time_step;

        const int max_threads = std::max(1, omp_get_max_threads());

        std::vector<ThreadContext> ctxs(max_threads);
        std::vector<std::vector<LayerGrads>> grads(max_threads);

        for (int t = 0; t < max_threads; ++t) {
            ctxs[t].resize(layers, input_size);
            grads[t].resize(layers.size());
            for (size_t i = 0; i < layers.size(); ++i) {
                grads[t][i].resize(layers[i]->W.size(), layers[i]->b.size());
            }
        }

        // Nulování gradientů (na začátku každého batch callu)
        for (int t = 0; t < max_threads; ++t) {
            for (size_t i = 0; i < layers.size(); ++i) grads[t][i].zero();
        }

        double total_sse = 0.0; // suma čtverců chyb (SSE)

        // Paralelní zpracování vzorků
#pragma omp parallel reduction(+:total_sse)
        {
            const int tid = omp_get_thread_num();
            ThreadContext& ctx = ctxs[tid];
            auto& tg = grads[tid];

#pragma omp for
            for (int b = 0; b < batch; ++b) {
                const double* x_ptr = in + (size_t)b * (size_t)in_len;
                const double* t_ptr = tgt + (size_t)b * (size_t)tgt_len;

                // 1) Forward
                std::memcpy(ctx.X0.data(), x_ptr, sizeof(double) * input_size);

                for (size_t i = 0; i < layers.size(); ++i) {
                    const double* x_in =
                        (i == 0) ? ctx.X0.data() : ctx.A[i - 1].data();
                    layers[i]->forward(x_in, ctx.Z[i].data(), ctx.A[i].data());
                }

                // 2) Loss + delta na výstupu:
                // d/dy ( (1/NK) * sum (y-t)^2 ) = 2*(y-t)/(N*K)
                // My akumulujeme SSE a MSE vyrobíme až nakonec.
                const double inv_NK = 1.0 / ((double)batch * (double)output_size);

                // delta_curr = dL/da na výstupu
                ctx.ensure_delta_sizes(output_size, layers.back()->in_sz);
                for (size_t j = 0; j < output_size; ++j) {
                    const double err = ctx.A.back()[j] - t_ptr[j];
                    total_sse += err * err;
                    ctx.delta_curr[j] = 2.0 * err * inv_NK;
                }

                // 3) Backward (ping-pong delta_curr -> delta_prev)
                // delta_curr má dim out_sz vrstvy i
                for (int li = (int)layers.size() - 1; li >= 0; --li) {
                    const DenseLayer* L = layers[(size_t)li].get();
                    const size_t curr_out = L->out_sz;
                    const size_t curr_in = L->in_sz;

                    // připrav delta_prev (dim curr_in) bez alokace ve smyčce
                    if (ctx.delta_curr.size() != curr_out) {
                        // bezpečnostní fallback (nemělo by nastat)
                        ctx.delta_curr.assign(curr_out, 0.0);
                    }
                    if (ctx.delta_prev.size() != curr_in) {
                        ctx.delta_prev.assign(curr_in, 0.0);
                    }
                    else {
                        std::fill(ctx.delta_prev.begin(), ctx.delta_prev.end(), 0.0);
                    }

                    const double* x_in = (li == 0) ? ctx.X0.data() : ctx.A[(size_t)li - 1].data();
                    L->backward(ctx.delta_curr.data(),
                        x_in,
                        ctx.Z[(size_t)li].data(),
                        ctx.A[(size_t)li].data(),
                        ctx.delta_prev.data(),
                        tg[(size_t)li].gW.data(),
                        tg[(size_t)li].gb.data());

                    // swap: delta_curr <- delta_prev (bez alokace)
                    ctx.delta_curr.resize(curr_in);
                    std::memcpy(ctx.delta_curr.data(), ctx.delta_prev.data(), sizeof(double) * curr_in);
                }
            }
        } // omp parallel

        // Redukce gradientů (sekvenčně, deterministické a jednoduché)
        for (int t = 1; t < max_threads; ++t) {
            for (size_t i = 0; i < layers.size(); ++i) {
                grads[0][i].add_inplace(grads[t][i]);
            }
        }

        // Update
        for (size_t i = 0; i < layers.size(); ++i) {
            layers[i]->update(grads[0][i].gW, grads[0][i].gb, hp, time_step);
        }

        if (mean_mse) {
            // total_sse už je suma přes batch i output, dělíme (batch*output) -> skutečné MSE
            *mean_mse = total_sse / ((double)batch * (double)output_size);
        }
        return true;
    }

    // -----------------------------------------------------------------------
    // Save/Load (binární, robustní)
    // -----------------------------------------------------------------------
    static bool write_exact(FILE* f, const void* p, size_t sz) {
        return (std::fwrite(p, 1, sz, f) == sz);
    }
    static bool read_exact(FILE* f, void* p, size_t sz) {
        return (std::fread(p, 1, sz, f) == sz);
    }

    bool save(const wchar_t* filename) const {
        if (!filename || !*filename) return false;
        FILE* f = nullptr;
        if (_wfopen_s(&f, filename, L"wb") != 0 || !f) return false;

        const uint32_t magic = 0x4E4E3031; // "NN01"
        const uint32_t version = 2;
        const uint32_t cnt = (uint32_t)layers.size();
        const uint64_t ts = time_step;

        bool ok = true;
        ok = ok && write_exact(f, &magic, sizeof(magic));
        ok = ok && write_exact(f, &version, sizeof(version));
        ok = ok && write_exact(f, &cnt, sizeof(cnt));
        ok = ok && write_exact(f, &ts, sizeof(ts));

        for (const auto& L : layers) {
            const uint64_t d_in = (uint64_t)L->in_sz;
            const uint64_t d_out = (uint64_t)L->out_sz;
            const int32_t  act = (int32_t)L->act;

            ok = ok && write_exact(f, &d_in, sizeof(d_in));
            ok = ok && write_exact(f, &d_out, sizeof(d_out));
            ok = ok && write_exact(f, &act, sizeof(act));

            ok = ok && write_exact(f, L->W.data(), sizeof(double) * L->W.size());
            ok = ok && write_exact(f, L->b.data(), sizeof(double) * L->b.size());

            if (!ok) break;
        }

        std::fclose(f);
        return ok;
    }

    bool load(const wchar_t* filename) {
        if (!filename || !*filename) return false;
        FILE* f = nullptr;
        if (_wfopen_s(&f, filename, L"rb") != 0 || !f) return false;

        uint32_t magic = 0, version = 0, cnt = 0;
        uint64_t ts = 0;

        bool ok = true;
        ok = ok && read_exact(f, &magic, sizeof(magic));
        ok = ok && (magic == 0x4E4E3031);
        ok = ok && read_exact(f, &version, sizeof(version));
        ok = ok && read_exact(f, &cnt, sizeof(cnt));
        ok = ok && read_exact(f, &ts, sizeof(ts));

        if (!ok) { std::fclose(f); return false; }

        // načteme do dočasné sítě a až pak swap (aby při chybě nezůstala polomrtvola)
        std::vector<std::unique_ptr<DenseLayer>> new_layers;
        new_layers.reserve(cnt);

        size_t new_input = 0, new_output = 0;

        for (uint32_t i = 0; i < cnt; ++i) {
            uint64_t d_in = 0, d_out = 0;
            int32_t act_code = 0;

            ok = ok && read_exact(f, &d_in, sizeof(d_in));
            ok = ok && read_exact(f, &d_out, sizeof(d_out));
            ok = ok && read_exact(f, &act_code, sizeof(act_code));
            if (!ok) break;

            if (d_in == 0 || d_out == 0) { ok = false; break; }
            const ActKind act = (ActKind)act_code;

            if (i == 0) new_input = (size_t)d_in;
            else {
                if (new_layers.back()->out_sz != (size_t)d_in) { ok = false; break; }
            }

            auto L = std::make_unique<DenseLayer>((size_t)d_in, (size_t)d_out, act);

            ok = ok && read_exact(f, L->W.data(), sizeof(double) * L->W.size());
            ok = ok && read_exact(f, L->b.data(), sizeof(double) * L->b.size());
            if (!ok) break;

            L->adam_W.reset();
            L->adam_b.reset();

            new_output = (size_t)d_out;
            new_layers.emplace_back(std::move(L));
        }

        std::fclose(f);

        if (!ok || new_layers.size() != (size_t)cnt) return false;

        layers.swap(new_layers);
        input_size = new_input;
        output_size = new_output;
        time_step = ts;
        return true;
    }
};

// ---------------------------------------------------------------------------
// UI (Win32 GDI) — jednoduchý MSE monitor, robustnější životní cyklus
// ---------------------------------------------------------------------------
namespace ui {
    struct MSEState {
        std::atomic<bool> running{ false };
        HANDLE thread{ nullptr };
        DWORD tid{ 0 };
        HWND hwnd{ nullptr };

        std::mutex mtx;
        std::deque<double> data;

        size_t max_points = 1000;
        bool autoscale = true;
        double y_min = 0.0, y_max = 1.0;

        std::atomic<bool> class_registered{ false };
    };

    static MSEState g;
    static const wchar_t* kClass = L"NNMSEWindowClass";
    static const wchar_t* kTitle = L"MSE Monitor";

    static void DrawMSEGraph(HDC hdc, const RECT& rc) {
        FillRect(hdc, &rc, (HBRUSH)(COLOR_WINDOW + 1));

        std::deque<double> local;
        {
            std::lock_guard<std::mutex> lk(g.mtx);
            local = g.data;
        }
        if (local.empty()) return;

        double vmin = 0.0, vmax = 1.0;
        if (g.autoscale) {
            vmin = std::numeric_limits<double>::infinity();
            vmax = -std::numeric_limits<double>::infinity();
            for (double v : local) {
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
        }
        else {
            vmin = g.y_min;
            vmax = g.y_max;
        }
        if (!(vmin < vmax)) vmax = vmin + 1.0;

        const int W = (rc.right - rc.left);
        const int H = (rc.bottom - rc.top);
        const int N = (int)local.size();
        if (W <= 1 || H <= 1 || N < 2) return;

        HPEN pen = CreatePen(PS_SOLID, 2, RGB(0, 120, 215));
        HGDIOBJ old = SelectObject(hdc, pen);

        for (int i = 0; i < N - 1; ++i) {
            const int x1 = rc.left + (i * W) / (N - 1);
            const int x2 = rc.left + ((i + 1) * W) / (N - 1);

            const double u1 = (local[i] - vmin) / (vmax - vmin);
            const double u2 = (local[i + 1] - vmin) / (vmax - vmin);

            const int y1 = rc.top + (H - 1) - (int)(u1 * (H - 1));
            const int y2 = rc.top + (H - 1) - (int)(u2 * (H - 1));

            MoveToEx(hdc, x1, y1, NULL);
            LineTo(hdc, x2, y2);
        }

        SelectObject(hdc, old);
        DeleteObject(pen);

        wchar_t buf[128];
        swprintf_s(buf, L"MSE: %.6g", local.back());
        TextOutW(hdc, rc.left + 6, rc.top + 6, buf, (int)wcslen(buf));
    }

    static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
        switch (m) {
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC dc = BeginPaint(h, &ps);
            RECT r; GetClientRect(h, &r);
            DrawMSEGraph(dc, r);
            EndPaint(h, &ps);
            return 0;
        }
        case WM_TIMER:
            if (g.running.load(std::memory_order_relaxed)) {
                InvalidateRect(h, NULL, FALSE);
            }
            return 0;

        case WM_CLOSE:
            DestroyWindow(h);
            return 0;

        case WM_DESTROY:
            g.hwnd = nullptr;
            PostQuitMessage(0);
            return 0;
        }
        return DefWindowProcW(h, m, w, l);
    }

    static bool EnsureClassRegistered() {
        if (g.class_registered.load(std::memory_order_acquire)) return true;

        WNDCLASSW wc{};
        wc.lpfnWndProc = WndProc;
        wc.hInstance = g_hInst;
        wc.lpszClassName = kClass;
        wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);

        if (!RegisterClassW(&wc)) {
            // pokud už existuje z dřívějška, bereme jako OK
            if (GetLastError() != ERROR_CLASS_ALREADY_EXISTS) return false;
        }
        g.class_registered.store(true, std::memory_order_release);
        return true;
    }

    static DWORD WINAPI ThreadProc(LPVOID) {
        if (!EnsureClassRegistered()) {
            g.running.store(false, std::memory_order_release);
            return 0;
        }

        g.hwnd = CreateWindowExW(
            WS_EX_TOPMOST | WS_EX_TOOLWINDOW,
            kClass, kTitle,
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            100, 100, 520, 320,
            0, 0, g_hInst, 0
        );

        if (!g.hwnd) {
            g.running.store(false, std::memory_order_release);
            return 0;
        }

        SetTimer(g.hwnd, 1, 100, NULL);

        MSG msg;
        while (GetMessageW(&msg, 0, 0, 0) > 0) {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }

        return 0;
    }

    void show(int s) {
        if (s) {
            if (g.running.load(std::memory_order_acquire)) return;
            g.running.store(true, std::memory_order_release);

            g.thread = CreateThread(nullptr, 0, ThreadProc, nullptr, 0, &g.tid);
            if (!g.thread) {
                g.running.store(false, std::memory_order_release);
            }
        }
        else {
            if (g.hwnd) {
                PostMessageW(g.hwnd, WM_CLOSE, 0, 0);
            }
        }
    }

    void push(double v) {
        std::lock_guard<std::mutex> lk(g.mtx);
        g.data.push_back(v);
        while (g.data.size() > g.max_points) g.data.pop_front();
    }

    void clear() {
        std::lock_guard<std::mutex> lk(g.mtx);
        g.data.clear();
    }

    void set_scale(int e, double mn, double mx) {
        std::lock_guard<std::mutex> lk(g.mtx);
        g.autoscale = (e != 0);
        g.y_min = mn;
        g.y_max = mx;
    }

    void set_pts(int n) {
        if (n <= 0) return;
        std::lock_guard<std::mutex> lk(g.mtx);
        g.max_points = (size_t)n;
    }

    void shutdown() {
        // jemný shutdown, bez TerminateThread
        if (g.hwnd) PostMessageW(g.hwnd, WM_CLOSE, 0, 0);

        if (g.thread) {
            WaitForSingleObject(g.thread, 5000);
            CloseHandle(g.thread);
            g.thread = nullptr;
        }
        g.running.store(false, std::memory_order_release);
        g.hwnd = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Instance manager
// ---------------------------------------------------------------------------
static std::unordered_map<int, std::unique_ptr<NeuralNetwork>> g_nn;
static std::mutex g_nn_mtx;
static int g_nn_next = 1;

static int nn_alloc() {
    try {
        std::lock_guard<std::mutex> lk(g_nn_mtx);
        const int h = g_nn_next++;
        g_nn[h] = std::make_unique<NeuralNetwork>();
        return h;
    }
    catch (...) {
        return 0;
    }
}

static NeuralNetwork* nn_get(int h) {
    std::lock_guard<std::mutex> lk(g_nn_mtx);
    auto it = g_nn.find(h);
    return (it != g_nn.end()) ? it->second.get() : nullptr;
}

static void nn_free(int h) {
    std::lock_guard<std::mutex> lk(g_nn_mtx);
    g_nn.erase(h);
}

static void nn_clear_all() {
    std::lock_guard<std::mutex> lk(g_nn_mtx);
    g_nn.clear();
}

// ---------------------------------------------------------------------------
// DLL EXPORTS (API zachováno)
// ---------------------------------------------------------------------------
DLL_EXTERN int  DLL_CALL NN_Create() {
    return nn_alloc();
}

DLL_EXTERN void DLL_CALL NN_Free(int h) {
    try { nn_free(h); }
    catch (...) {}
}

DLL_EXTERN bool DLL_CALL NN_AddDense(int h, int i, int o, int a) {
    try {
        auto* n = nn_get(h);
        return n ? n->add_dense((size_t)i, (size_t)o, (ActKind)a) : false;
    } DLL_CATCH_ALL(false)
}

DLL_EXTERN bool DLL_CALL NN_TrainBatch(int h,
    const double* in, int b, int il,
    const double* t, int tl,
    double lr, double* mse)
{
    try {
        auto* n = nn_get(h);
        return n ? n->train_batch(in, b, il, t, tl, lr, mse) : false;
    } DLL_CATCH_ALL(false)
}

DLL_EXTERN bool DLL_CALL NN_Forward(int h, const double* in, int il, double* out, int ol) {
    try {
        auto* n = nn_get(h);
        return n ? n->forward(in, il, out, ol) : false;
    } DLL_CATCH_ALL(false)
}

DLL_EXTERN bool DLL_CALL NN_Save(int h, const wchar_t* p) {
    try {
        auto* n = nn_get(h);
        return n ? n->save(p) : false;
    } DLL_CATCH_ALL(false)
}

DLL_EXTERN bool DLL_CALL NN_Load(int h, const wchar_t* p) {
    try {
        auto* n = nn_get(h);
        return n ? n->load(p) : false;
    } DLL_CATCH_ALL(false)
}

DLL_EXTERN int DLL_CALL NN_InputSize(int h) {
    try {
        auto* n = nn_get(h);
        return n ? (int)n->in_size() : 0;
    } DLL_CATCH_ALL(0)
}

DLL_EXTERN int DLL_CALL NN_OutputSize(int h) {
    try {
        auto* n = nn_get(h);
        return n ? (int)n->out_size() : 0;
    } DLL_CATCH_ALL(0)
}

DLL_EXTERN void DLL_CALL NN_SetSeed(unsigned int s) {
    try { rng::seed((uint64_t)s); }
    catch (...) {}
}

// UI
DLL_EXTERN void DLL_CALL NN_MSE_Push(double m) { try { ui::push(m); } catch (...) {} }
DLL_EXTERN void DLL_CALL NN_MSE_Show(int s) { try { ui::show(s); } catch (...) {} }
DLL_EXTERN void DLL_CALL NN_MSE_Clear() { try { ui::clear(); } catch (...) {} }
DLL_EXTERN void DLL_CALL NN_MSE_SetMaxPoints(int n) { try { ui::set_pts(n); } catch (...) {} }
DLL_EXTERN void DLL_CALL NN_MSE_SetAutoScale(int e, double mn, double mx) { try { ui::set_scale(e, mn, mx); } catch (...) {} }

// Cleanup
DLL_EXTERN void DLL_CALL NN_GlobalCleanup() {
    try {
        ui::shutdown();
        nn_clear_all();
    }
    catch (...) {}
}

// ---------------------------------------------------------------------------
// DllMain
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE h, DWORD r, LPVOID) {
    if (r == DLL_PROCESS_ATTACH) {
        g_hInst = (HINSTANCE)h;
        DisableThreadLibraryCalls(h);
    }
    else if (r == DLL_PROCESS_DETACH) {
        // obrana proti visícímu UI threadu při unloadu
        ui::shutdown();
        nn_clear_all();
    }
    return TRUE;
}
