# NNMQL5FIX

**NNMQL5FIX — Lightweight MLP DLL for MetaTrader 5 (x64, MSVC)**

Author: **Remind — Tomáš Bělák**  
License: **MIT-like** (use freely, but credit the author)

A small, numerically stable MLP library as a DLL for MQL5.  
No external dependencies, pure C API, 64-bit, MSVC.  

Features:  
- forward pass  
- training (SGD)  
- mini-batch wrappers  
- weight access  

---

## Features

- Stable sigmoid (no overflow/underflow)  
- Accurate dot products: Neumaier compensation + FMA  
- Compensated summation for MSE and batch MSE  
- Gradient clipping and conservative numerics  
- Pure C exports (no STL across DLL boundary)  
- Build with `/MT` (static CRT) → no VC++ Redistributable required  

**Note:** This version is sequential (parallelism disabled).

---

## Build (Visual Studio 2022, x64)

### Precompiled Headers (PCH)

**pch.h:**
```cpp
#pragma once
#include <windows.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cstdlib>
```

**pch.cpp:**
```cpp
#include "pch.h"
```

**Settings:**
- `pch.cpp` → C/C++ → Precompiled Headers → **Create (/Yc)**, Header: `pch.h`  
- Other `.cpp` files → **Use (/Yu)**, Header: `pch.h`  
- Each `.cpp` must start with:  
  ```cpp
  #include "pch.h"
  ```

### Static CRT
- Release: `/MT`  
- Debug: `/MTd`  

### Target
- Platform: **x64**  
- Rebuild project  

### Verification
```bash
dumpbin /dependents NNMQL5FIX.dll
```
The list must not contain `VCRUNTIME*`, `UCRTBASE.dll`.

---

## API (exports)

| Function | Description |
|----------|-------------|
| `int NN_Create()` | Creates a network instance, returns handle `h > 0`. |
| `void NN_Free(int h)` | Frees an instance. |
| `bool NN_AddDense(int h, int inSz, int outSz, int act)` | Adds a dense layer. Activations: `0=SIGMOID`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG`. |
| `int NN_InputSize(int h)` | Network input size. |
| `int NN_OutputSize(int h)` | Network output size. |
| `bool NN_Forward(...)` | Forward pass for 1 sample. |
| `bool NN_ForwardBatch(...)` | Forward pass for a batch of samples (rows). |
| `bool NN_TrainOne(...)` | Train with 1 sample (SGD), returns MSE. |
| `bool NN_TrainBatch(...)` | Batch training (sequential, average MSE). |
| `bool NN_GetWeights(...)` | Reads weights/biases of layer *i*. |
| `bool NN_SetWeights(...)` | Writes weights/biases of layer *i*. |

**Batch format:**
- `in` size: `batch * in_len` (samples by rows)  
- `out` size: `batch * out_len`  
- `tgt` size: `batch * tgt_len`

---

## Usage in MQL5

### Import block
```mql
#import "NNMQL5FIX.dll"
int  NN_Create();
void NN_Free(int h);
bool NN_AddDense(int h, int inSz, int outSz, int act);
int  NN_InputSize(int h);
int  NN_OutputSize(int h);
bool NN_Forward(int h, const double &in[], int in_len, double &out[], int out_len);
bool NN_ForwardBatch(int h, const double &in[], int batch, int in_len, double &out[], int out_len);
bool NN_TrainOne(int h, const double &in[], int in_len, const double &tgt[], int tgt_len, double lr, double &mse);
bool NN_TrainBatch(int h, const double &in[], int batch, int in_len, const double &tgt[], int tgt_len, double lr, double &mean_mse);
bool NN_GetWeights(int h, int i, double &W[], int Wlen, double &b[], int blen);
bool NN_SetWeights(int h, int i, const double &W[], int Wlen, const double &b[], int blen);
#import
```

### Mini-example
```mql
int h = NN_Create();
NN_AddDense(h, 32, 64, 2); // TANH
NN_AddDense(h, 64, 1, 3);  // LINEAR

// forward pass
double x[32]; ArrayInitialize(x, 0.0);
double y[1];
NN_Forward(h, x, 32, y, 1);

// training mini-batch
const int B = 16;
double in[B*32], tgt[B*1];
double mean_mse = 0.0;
NN_TrainBatch(h, in, B, 32, tgt, 1, 0.001, mean_mse);

NN_Free(h);
```

Place DLL into `MQL5\Libraries\`.  
EA/indicator will run **without VC++ redistributable**.

---

## Numerical notes

- Sigmoid implemented stably (branches for `x ≥ 0` and `x < 0`)  
- Dot products: `std::fma` + Neumaier compensation  
- MSE and batch MSE: compensated summations  
- Training: SGD with **gradient clipping (±5.0)**  

---

## Limitations and recommendations

- Sequential execution (no threads); each handle is independent  
- Do not pass STL objects across the DLL boundary (API is pure C, safe)  
- `in_len` and `out_len` must match network topology, otherwise functions return `false`  
- Activations: `0=SIG`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG`  

---

## Repository structure

```
NNMQL5FIX/
├─ src/
│  ├─ dllmain.cpp
│  ├─ pch.h
│  └─ pch.cpp
├─ build/           # outputs (DLL, LIB, PDB)
├─ examples/
│  └─ MQL5/         # import examples in indicator/EA
└─ README.md
```

---

## License

MIT-like spirit — use freely, but please credit **Remind — Tomáš Bělák**.  
No warranty; software is provided **“as is”**.

---

## Acknowledgments

Thanks to the MQL5 community and everyone who loves clean numerics and simple APIs.  
Pull requests welcome.
