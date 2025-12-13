// ============================================================================
//  dllmain.cpp — High-Performance MLP DLL for MQL5 (x64, MSVC)
//  --------------------------------------------------------------------------
//  VERSION: 3.2 (Fixed Loader Lock Crash)
//  FEATURES:
//  1. Optimization: AdamW.
//  2. Parallelism: OpenMP multithreading.
//  3. Precision: Neumaier summation, FMA.
//  4. Architecture: Stateless layers.
//  5. Persistence: Binary Save/Load.
//  6. Tools: Built-in GDI+ MSE Graphing.
//  7. Safety: Explicit Cleanup to avoid Loader Lock deadlocks.
//
//  (c) 2025, Remind — Optimized Architecture
// ============================================================================

#include "pch.h" // PCH must be first

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
#include <omp.h> // Requires /openmp in Visual Studio

// ---------------------------------------------------------------------------
// Macros & Constants
// ---------------------------------------------------------------------------
#ifndef DLL_EXTERN
#define DLL_EXTERN extern "C" __declspec(dllexport)
#endif
#ifndef DLL_CALL
#define DLL_CALL __cdecl
#endif

// Hyperparameters for AdamW
struct HyperParams {
    double lr = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
    double weight_decay = 0.01;
};

static HINSTANCE g_hInst = nullptr;

// ---------------------------------------------------------------------------
// RNG & Utils
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

static std::wstring utf8_to_wide(const char* s) {
    if (!s) return L"";
    int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if (n <= 0) return L"<invalid utf8>";
    std::wstring w(static_cast<size_t>(n - 1), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    return w;
}

// ---------------------------------------------------------------------------
// Precise Math
// ---------------------------------------------------------------------------
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
    inline double tanh_act(double x) { return std::tanh(x); }

    // Neumaier Summation
    inline double neumaier_add(double sum, double val, double& comp) {
        double t = sum + val;
        if (std::abs(sum) >= std::abs(val)) {
            comp += (sum - t) + val;
        }
        else {
            comp += (val - t) + sum;
        }
        return t;
    }

    inline double dot_product(const double* a, const double* b, size_t n) {
        double sum = 0.0, c = 0.0;
        for (size_t i = 0; i < n; ++i) {
            double prod = std::fma(a[i], b[i], 0.0);
            sum = neumaier_add(sum, prod, c);
        }
        return sum + c;
    }
}

// ---------------------------------------------------------------------------
// Optimizer (AdamW) Helper
// ---------------------------------------------------------------------------
struct AdamState {
    std::vector<double> m; // 1st moment
    std::vector<double> v; // 2nd moment

    void resize(size_t n) {
        m.assign(n, 0.0);
        v.assign(n, 0.0);
    }

    void reset() {
        std::fill(m.begin(), m.end(), 0.0);
        std::fill(v.begin(), v.end(), 0.0);
    }

    void update(std::vector<double>& params, const std::vector<double>& grads, const HyperParams& hp, uint64_t t) {
        double bias_correction1 = 1.0 - std::pow(hp.beta1, (double)t);
        double bias_correction2 = 1.0 - std::pow(hp.beta2, (double)t);
        const size_t n = params.size();

        // Vectorized loop
        for (size_t i = 0; i < n; ++i) {
            double g = grads[i];
            params[i] -= hp.lr * hp.weight_decay * params[i]; // Decoupled WD

            m[i] = hp.beta1 * m[i] + (1.0 - hp.beta1) * g;
            v[i] = hp.beta2 * v[i] + (1.0 - hp.beta2) * g * g;

            double m_hat = m[i] / bias_correction1;
            double v_hat = v[i] / bias_correction2;

            params[i] -= hp.lr * m_hat / (std::sqrt(v_hat) + hp.eps);
        }
    }
};

// ---------------------------------------------------------------------------
// Enums & Structs
// ---------------------------------------------------------------------------
enum class ActKind : int { SIGMOID = 0, RELU = 1, TANH = 2, LINEAR = 3, SYM_SIG = 4 };

struct Activation {
    static double f(ActKind k, double x) {
        switch (k) {
        case ActKind::SIGMOID: return precise::sigmoid(x);
        case ActKind::RELU:    return x > 0.0 ? x : 0.0;
        case ActKind::TANH:    return std::tanh(x);
        case ActKind::SYM_SIG: return 2.0 * precise::sigmoid(x) - 1.0;
        default: return x;
        }
    }
    static double df(ActKind k, double y, double x) {
        (void)x;
        switch (k) {
        case ActKind::SIGMOID: return y * (1.0 - y);
        case ActKind::RELU:    return x > 0.0 ? 1.0 : 0.0;
        case ActKind::TANH:    return 1.0 - y * y;
        case ActKind::SYM_SIG: return 0.5 * (1.0 - y * y);
        default: return 1.0;
        }
    }
};

// ---------------------------------------------------------------------------
// Dense Layer
// ---------------------------------------------------------------------------
struct DenseLayer {
    size_t in_sz, out_sz;
    std::vector<double> W; // Row-major: [out][in]
    std::vector<double> b;
    ActKind act;
    AdamState adam_W, adam_b;

    DenseLayer(size_t in, size_t out, ActKind k) : in_sz(in), out_sz(out), act(k) {
        W.resize(in * out);
        b.resize(out, 0.0);
        adam_W.resize(W.size());
        adam_b.resize(b.size());
        double scale = (k == ActKind::RELU) ? std::sqrt(2.0 / in) : std::sqrt(1.0 / in);
        for (double& w : W) w = ((std::rand() / (double)RAND_MAX) * 2.0 - 1.0) * scale;
    }

    void forward(const std::vector<double>& input, std::vector<double>& z, std::vector<double>& a) const {
        for (size_t o = 0; o < out_sz; ++o) {
            double val = precise::dot_product(&W[o * in_sz], input.data(), in_sz);
            val += b[o];
            z[o] = val;
            a[o] = Activation::f(act, val);
        }
    }

    void backward(const std::vector<double>& dL_da, const std::vector<double>& input,
        const std::vector<double>& z, const std::vector<double>& a,
        std::vector<double>& dL_dx_prev,
        std::vector<double>& grad_W, std::vector<double>& grad_b) const
    {
        std::vector<double> dL_dz(out_sz);
        for (size_t o = 0; o < out_sz; ++o) {
            dL_dz[o] = dL_da[o] * Activation::df(act, a[o], z[o]);
            grad_b[o] += dL_dz[o];
        }
        for (size_t o = 0; o < out_sz; ++o) {
            double dz = dL_dz[o];
            double* row_grad = &grad_W[o * in_sz];
            const double* in_ptr = input.data();
            for (size_t i = 0; i < in_sz; ++i) row_grad[i] += dz * in_ptr[i];
        }
        std::fill(dL_dx_prev.begin(), dL_dx_prev.end(), 0.0);
        for (size_t o = 0; o < out_sz; ++o) {
            double dz = dL_dz[o];
            const double* w_row = &W[o * in_sz];
            for (size_t i = 0; i < in_sz; ++i) dL_dx_prev[i] += w_row[i] * dz;
        }
    }

    void update(const std::vector<double>& grad_W, const std::vector<double>& grad_b,
        const HyperParams& hp, uint64_t t)
    {
        adam_W.update(W, grad_W, hp, t);
        adam_b.update(b, grad_b, hp, t);
    }
};

// ---------------------------------------------------------------------------
// Neural Network (MLP)
// ---------------------------------------------------------------------------
class NeuralNetwork {
    std::vector<std::unique_ptr<DenseLayer>> layers;
    size_t input_size{ 0 }, output_size{ 0 };
    uint64_t time_step{ 0 };

    struct ThreadContext {
        std::vector<std::vector<double>> inputs, Z, A;
        std::vector<double> delta;
        void resize(const std::vector<std::unique_ptr<DenseLayer>>& Ls, size_t in_sz) {
            inputs.resize(Ls.size()); Z.resize(Ls.size()); A.resize(Ls.size());
            size_t max_dim = in_sz;
            for (size_t i = 0; i < Ls.size(); ++i) {
                inputs[i].resize(Ls[i]->in_sz); Z[i].resize(Ls[i]->out_sz); A[i].resize(Ls[i]->out_sz);
                if (Ls[i]->in_sz > max_dim) max_dim = Ls[i]->in_sz;
                if (Ls[i]->out_sz > max_dim) max_dim = Ls[i]->out_sz;
            }
            delta.resize(max_dim);
        }
    };

    struct LayerGrads {
        std::vector<double> gW, gb;
        void resize(size_t w_sz, size_t b_sz) { gW.assign(w_sz, 0.0); gb.assign(b_sz, 0.0); }
        void add(const LayerGrads& other) {
            for (size_t i = 0; i < gW.size(); ++i) gW[i] += other.gW[i];
            for (size_t i = 0; i < gb.size(); ++i) gb[i] += other.gb[i];
        }
    };

public:
    bool add_dense(size_t in_sz, size_t out_sz, ActKind k) {
        if (layers.empty()) input_size = in_sz;
        else if (layers.back()->out_sz != in_sz) return false;
        layers.emplace_back(std::make_unique<DenseLayer>(in_sz, out_sz, k));
        output_size = out_sz;
        return true;
    }

    size_t in_size() const { return input_size; }
    size_t out_size() const { return output_size; }

    bool forward(const double* in, int in_len, double* out, int out_len) {
        if ((int)input_size != in_len || (int)output_size != out_len || layers.empty()) return false;
        std::vector<double> x(in, in + in_len), z_dummy, a_next;
        for (auto& L : layers) {
            z_dummy.resize(L->out_sz); a_next.resize(L->out_sz);
            L->forward(x, z_dummy, a_next); x = a_next;
        }
        for (int i = 0; i < out_len; ++i) out[i] = x[i];
        return true;
    }

    bool train_batch(const double* in, int batch, int in_len, const double* tgt, int tgt_len, double lr, double* mean_mse) {
        if (layers.empty() || batch <= 0) return false;
        time_step++;
        HyperParams hp; hp.lr = lr;

        int max_threads = omp_get_max_threads();
        std::vector<ThreadContext> contexts(max_threads);
        std::vector<std::vector<LayerGrads>> thread_grads(max_threads);

        for (int t = 0; t < max_threads; ++t) {
            contexts[t].resize(layers, input_size);
            thread_grads[t].resize(layers.size());
            for (size_t i = 0; i < layers.size(); ++i) thread_grads[t][i].resize(layers[i]->W.size(), layers[i]->b.size());
        }

        double total_mse = 0.0;
#pragma omp parallel 
        {
            int tid = omp_get_thread_num();
            ThreadContext& ctx = contexts[tid];
            auto& t_grads = thread_grads[tid];
            double local_mse = 0.0;

#pragma omp for
            for (int b = 0; b < batch; ++b) {
                const double* x_ptr = in + (size_t)b * in_len;
                const double* t_ptr = tgt + (size_t)b * tgt_len;
                ctx.inputs[0].assign(x_ptr, x_ptr + in_len);
                for (size_t i = 0; i < layers.size(); ++i) {
                    const std::vector<double>& curr_in = (i == 0) ? ctx.inputs[0] : ctx.A[i - 1];
                    layers[i]->forward(curr_in, ctx.Z[i], ctx.A[i]);
                }
                const std::vector<double>& y = ctx.A.back();
                for (size_t j = 0; j < output_size; ++j) {
                    double err = y[j] - t_ptr[j];
                    local_mse += err * err;
                    ctx.delta[j] = 2.0 * err / batch;
                }
                for (int i = (int)layers.size() - 1; i >= 0; --i) {
                    const std::vector<double>& curr_in = (i == 0) ? ctx.inputs[0] : ctx.A[i - 1];
                    std::vector<double> next_delta(layers[i]->in_sz);
                    layers[i]->backward(ctx.delta, curr_in, ctx.Z[i], ctx.A[i], next_delta, t_grads[i].gW, t_grads[i].gb);
                    ctx.delta = next_delta;
                }
            }
#pragma omp atomic
            total_mse += local_mse;
        }

        for (int t = 1; t < max_threads; ++t) {
            for (size_t i = 0; i < layers.size(); ++i) thread_grads[0][i].add(thread_grads[t][i]);
        }
        for (size_t i = 0; i < layers.size(); ++i) layers[i]->update(thread_grads[0][i].gW, thread_grads[0][i].gb, hp, time_step);
        if (mean_mse) *mean_mse = total_mse / batch;
        return true;
    }

    bool save(const wchar_t* filename) const {
        FILE* f = nullptr;
        if (_wfopen_s(&f, filename, L"wb") != 0 || !f) return false;
        uint32_t magic = 0x4E4E3031, version = 1, cnt = (uint32_t)layers.size();
        fwrite(&magic, sizeof(magic), 1, f); fwrite(&version, sizeof(version), 1, f); fwrite(&cnt, sizeof(cnt), 1, f);
        for (const auto& L : layers) {
            uint64_t d_in = (uint64_t)L->in_sz, d_out = (uint64_t)L->out_sz;
            int32_t act_code = (int32_t)L->act;
            fwrite(&d_in, sizeof(d_in), 1, f); fwrite(&d_out, sizeof(d_out), 1, f); fwrite(&act_code, sizeof(act_code), 1, f);
            fwrite(L->W.data(), sizeof(double), L->W.size(), f); fwrite(L->b.data(), sizeof(double), L->b.size(), f);
        }
        fclose(f); return true;
    }

    bool load(const wchar_t* filename) {
        FILE* f = nullptr;
        if (_wfopen_s(&f, filename, L"rb") != 0 || !f) return false;
        uint32_t magic = 0, version = 0, cnt = 0;
        if (fread(&magic, sizeof(magic), 1, f) != 1 || magic != 0x4E4E3031) { fclose(f); return false; }
        if (fread(&version, sizeof(version), 1, f) != 1 || fread(&cnt, sizeof(cnt), 1, f) != 1) { fclose(f); return false; }
        layers.clear(); input_size = 0; output_size = 0; time_step = 0;
        for (uint32_t i = 0; i < cnt; ++i) {
            uint64_t d_in, d_out; int32_t act_code;
            if (fread(&d_in, sizeof(d_in), 1, f) != 1 || fread(&d_out, sizeof(d_out), 1, f) != 1 || fread(&act_code, sizeof(act_code), 1, f) != 1) break;
            if (!add_dense((size_t)d_in, (size_t)d_out, (ActKind)act_code)) { fclose(f); return false; }
            auto& L = layers.back();
            fread(L->W.data(), sizeof(double), L->W.size(), f); fread(L->b.data(), sizeof(double), L->b.size(), f);
            L->adam_W.reset(); L->adam_b.reset();
        }
        fclose(f); return true;
    }
};

// ---------------------------------------------------------------------------
// Visualization (UI) - GDI+ Graph
// ---------------------------------------------------------------------------
namespace ui {
    struct MSEState {
        std::atomic<bool> running{ false };
        std::atomic<bool> visible{ false };
        HANDLE thread{ nullptr };
        DWORD tid{ 0 };
        HWND hwnd{ nullptr };
        std::mutex mtx;
        std::deque<double> data;
        size_t max_points = 1000;
        bool autoscale = true;
        double y_min = 0.0, y_max = 1.0;
    };
    static MSEState g;
    static const wchar_t* kClass = L"NNMSEWindowClass";
    static const wchar_t* kTitle = L"MSE Monitor (GDI+)";

    static void DrawMSEGraph(HDC hdc, RECT rc) {
        HBRUSH bg = (HBRUSH)(COLOR_WINDOW + 1); FillRect(hdc, &rc, bg);
        std::deque<double> local;
        { std::lock_guard<std::mutex> lk(g.mtx); local = g.data; }
        if (local.empty()) return;
        double vmin = 1e9, vmax = -1e9;
        if (g.autoscale) { for (double v : local) { if (v < vmin) vmin = v; if (v > vmax) vmax = v; } }
        else { vmin = g.y_min; vmax = g.y_max; }
        if (vmin >= vmax) vmax = vmin + 1.0;
        HPEN lnPen = CreatePen(PS_SOLID, 2, RGB(0, 120, 215));
        HGDIOBJ old = SelectObject(hdc, lnPen);
        int N = (int)local.size(), W = rc.right, H = rc.bottom;
        for (int i = 0; i < N - 1; ++i) {
            int x1 = (i * W) / N; int x2 = ((i + 1) * W) / N;
            int y1 = H - (int)((local[i] - vmin) / (vmax - vmin) * H);
            int y2 = H - (int)((local[i + 1] - vmin) / (vmax - vmin) * H);
            MoveToEx(hdc, x1, y1, NULL); LineTo(hdc, x2, y2);
        }
        SelectObject(hdc, old); DeleteObject(lnPen);
        wchar_t buf[64]; swprintf_s(buf, L"MSE: %.6g", local.back());
        TextOutW(hdc, 5, 5, buf, (int)wcslen(buf));
    }

    static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
        if (m == WM_PAINT) { PAINTSTRUCT ps; HDC dc = BeginPaint(h, &ps); RECT r; GetClientRect(h, &r); DrawMSEGraph(dc, r); EndPaint(h, &ps); return 0; }
        if (m == WM_TIMER) InvalidateRect(h, NULL, FALSE);
        return DefWindowProc(h, m, w, l);
    }

    static DWORD WINAPI ThreadProc(LPVOID) {
        WNDCLASSW wc = { 0 }; wc.lpfnWndProc = WndProc; wc.hInstance = g_hInst; wc.lpszClassName = kClass; wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1); RegisterClassW(&wc);
        g.hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, kClass, kTitle, WS_OVERLAPPEDWINDOW | WS_VISIBLE, 100, 100, 500, 300, 0, 0, g_hInst, 0);
        SetTimer(g.hwnd, 1, 100, NULL);
        MSG msg; while (GetMessage(&msg, 0, 0, 0) > 0) { TranslateMessage(&msg); DispatchMessage(&msg); }
        // Clean up window class on exit
        UnregisterClassW(kClass, g_hInst);
        return 0;
    }

    void show(int s) {
        if (s && !g.running) { g.running = true; g.visible = true; g.thread = CreateThread(0, 0, ThreadProc, 0, 0, &g.tid); }
        else if (!s && g.hwnd) { PostMessage(g.hwnd, WM_CLOSE, 0, 0); g.running = false; }
    }
    void push(double v) { std::lock_guard<std::mutex> lk(g.mtx); g.data.push_back(v); if (g.data.size() > g.max_points) g.data.pop_front(); }
    void clear() { std::lock_guard<std::mutex> lk(g.mtx); g.data.clear(); }
    void set_scale(int e, double mn, double mx) { std::lock_guard<std::mutex> lk(g.mtx); g.autoscale = e; g.y_min = mn; g.y_max = mx; }
    void set_pts(int n) { std::lock_guard<std::mutex> lk(g.mtx); g.max_points = n; }

    // SAFE SHUTDOWN: To be called from exported Cleanup function, NOT DllMain
    void shutdown() {
        if (g.running) {
            PostThreadMessage(g.tid, WM_QUIT, 0, 0);
            if (g.thread) {
                WaitForSingleObject(g.thread, 1000); // Safe here, outside DllMain
                CloseHandle(g.thread);
                g.thread = nullptr;
            }
            g.running = false;
        }
    }
}

// ---------------------------------------------------------------------------
// Instance Manager - MLP
// ---------------------------------------------------------------------------
static std::unordered_map<int, std::unique_ptr<NeuralNetwork>> g_nn;
static std::mutex g_nn_mtx;
static int g_nn_next = 1;

int nn_alloc() { std::lock_guard<std::mutex> lk(g_nn_mtx); g_nn[g_nn_next] = std::make_unique<NeuralNetwork>(); return g_nn_next++; }
NeuralNetwork* nn_get(int h) { std::lock_guard<std::mutex> lk(g_nn_mtx); return g_nn.count(h) ? g_nn[h].get() : nullptr; }
void nn_free(int h) { std::lock_guard<std::mutex> lk(g_nn_mtx); g_nn.erase(h); }

// Helper to clear all resources safely
void nn_clear_all() {
    std::lock_guard<std::mutex> lk(g_nn_mtx);
    g_nn.clear();
}

// ---------------------------------------------------------------------------
// DLL EXPORTS
// ---------------------------------------------------------------------------

// --- MLP Exports ---
DLL_EXTERN int  DLL_CALL NN_Create() { return nn_alloc(); }
DLL_EXTERN void DLL_CALL NN_Free(int h) { nn_free(h); }
DLL_EXTERN bool DLL_CALL NN_AddDense(int h, int i, int o, int a) { auto* n = nn_get(h); return n ? n->add_dense(i, o, (ActKind)a) : false; }
DLL_EXTERN bool DLL_CALL NN_TrainBatch(int h, const double* in, int b, int il, const double* t, int tl, double lr, double* mse) {
    auto* n = nn_get(h); return n ? n->train_batch(in, b, il, t, tl, lr, mse) : false;
}
DLL_EXTERN bool DLL_CALL NN_Forward(int h, const double* in, int il, double* out, int ol) {
    auto* n = nn_get(h); return n ? n->forward(in, il, out, ol) : false;
}
DLL_EXTERN bool DLL_CALL NN_Save(int h, const wchar_t* p) { auto* n = nn_get(h); return n ? n->save(p) : false; }
DLL_EXTERN bool DLL_CALL NN_Load(int h, const wchar_t* p) { auto* n = nn_get(h); return n ? n->load(p) : false; }

// --- HELPERS ---
DLL_EXTERN int  DLL_CALL NN_InputSize(int h) { auto* n = nn_get(h); return n ? (int)n->in_size() : 0; }
DLL_EXTERN int  DLL_CALL NN_OutputSize(int h) { auto* n = nn_get(h); return n ? (int)n->out_size() : 0; }

DLL_EXTERN void DLL_CALL NN_SetSeed(unsigned int s) {
    std::srand(s);
    rng::seed(s);
}

DLL_EXTERN int DLL_CALL NN_GetLastError(int h, int* code, wchar_t* buf, int buf_len) {
    if (code) *code = 0;
    if (buf && buf_len > 0) {
        if (buf_len > 2) wcscpy_s(buf, buf_len, L"OK");
        else buf[0] = L'\0';
    }
    return 0;
}

// --- UI Exports ---
DLL_EXTERN void DLL_CALL NN_MSE_Push(double m) { ui::push(m); }
DLL_EXTERN void DLL_CALL NN_MSE_Show(int s) { ui::show(s); }
DLL_EXTERN void DLL_CALL NN_MSE_Clear() { ui::clear(); }
DLL_EXTERN void DLL_CALL NN_MSE_SetMaxPoints(int n) { ui::set_pts(n); }
DLL_EXTERN void DLL_CALL NN_MSE_SetAutoScale(int e, double mn, double mx) { ui::set_scale(e, mn, mx); }

// --- CLEANUP EXPORT (NEW) ---
// CRITICAL: Call this from MQL5 OnDeinit() to safely stop threads and free memory.
DLL_EXTERN void DLL_CALL NN_GlobalCleanup() {
    // 1. Stop UI Thread (safe here, not in DllMain)
    ui::shutdown();
    // 2. Clear all networks
    nn_clear_all();
}

// ---------------------------------------------------------------------------
// DllMain
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE h, DWORD r, LPVOID) {
    if (r == DLL_PROCESS_ATTACH) {
        g_hInst = (HINSTANCE)h;
        DisableThreadLibraryCalls(h);
    }
    // REMOVED: ui::shutdown() from DLL_PROCESS_DETACH to prevent Loader Lock deadlock.
    // Cleanup is now manual via NN_GlobalCleanup().
    return TRUE;
}