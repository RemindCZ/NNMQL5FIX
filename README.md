# NNMQL5FIX

**NNMQL5FIX — Lehká MLP DLL pro MetaTrader 5 (x64, MSVC)**

Autor: **Remind — Tomáš Bělák**  
Licence: **MIT-like** (užijte, ale uveďte autorství)

Malá, numericky stabilní MLP knihovna jako DLL pro MQL5.  
Bez externích závislostí, čisté C API, 64-bit, MSVC.  

Funkce:  
- dopředný průchod  
- učení (SGD)  
- mini-batch obálky  
- přístup k vahám  

---

## Vlastnosti

- Stabilní sigmoid (bez overflow/underflow)  
- Přesné dot-produkty: Neumaierova kompenzace + FMA  
- Kompenzované sumace MSE i batch MSE  
- Gradient clipping a konzervativní numerika  
- Čisté C exporty (žádné STL přes hranici DLL)  
- Build s `/MT` (statický CRT) → bez VC++ Redistributable  

**Pozn.:** Tato verze je sekvenční (paralelismus vypnutý).

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

**Nastavení:**
- `pch.cpp` → C/C++ → Precompiled Headers → **Create (/Yc)**, Header: `pch.h`  
- Ostatní `.cpp` → **Use (/Yu)**, Header: `pch.h`  
- Každý `.cpp` musí začínat:  
  ```cpp
  #include "pch.h"
  ```

### Statický CRT
- Release: `/MT`  
- Debug: `/MTd`  

### Cíl
- Platforma: **x64**  
- Rebuild projektu  

### Ověření
```bash
dumpbin /dependents NNMQL5FIX.dll
```
V seznamu nesmí být `VCRUNTIME*`, `UCRTBASE.dll`.

---

## API (exporty)

| Funkce | Popis |
|--------|-------|
| `int NN_Create()` | Vytvoří instanci sítě, vrací handle `h > 0`. |
| `void NN_Free(int h)` | Zruší instanci. |
| `bool NN_AddDense(int h, int inSz, int outSz, int act)` | Přidá dense vrstvu. Aktivace: `0=SIGMOID`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG`. |
| `int NN_InputSize(int h)` | Velikost vstupu sítě. |
| `int NN_OutputSize(int h)` | Velikost výstupu sítě. |
| `bool NN_Forward(...)` | Dopředný průchod pro 1 vzorek. |
| `bool NN_ForwardBatch(...)` | Dopředný průchod pro batch vzorků (řádky). |
| `bool NN_TrainOne(...)` | Učení 1 vzorku (SGD), vrací MSE. |
| `bool NN_TrainBatch(...)` | Učení po dávkách (sekvenční, průměrné MSE). |
| `bool NN_GetWeights(...)` | Načte váhy/biasy vrstvy *i*. |
| `bool NN_SetWeights(...)` | Zapíše váhy/biasy vrstvy *i*. |

**Formát batchů:**
- `in` má velikost `batch * in_len` (samples po řádcích)  
- `out` má velikost `batch * out_len`  
- `tgt` má velikost `batch * tgt_len`

---

## Použití v MQL5

### Import blok
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

### Mini-příklad
```mql
int h = NN_Create();
NN_AddDense(h, 32, 64, 2); // TANH
NN_AddDense(h, 64, 1, 3);  // LINEAR

// dopředný průchod
double x[32]; ArrayInitialize(x, 0.0);
double y[1];
NN_Forward(h, x, 32, y, 1);

// učení mini-batche
const int B = 16;
double in[B*32], tgt[B*1];
double mean_mse = 0.0;
NN_TrainBatch(h, in, B, 32, tgt, 1, 0.001, mean_mse);

NN_Free(h);
```

DLL umístěte do `MQL5\Libraries\`.  
EA/indikátor poběží **bez VC++ redistributable**.

---

## Numerické poznámky

- Sigmoid je implementován stabilně (větve pro `x ≥ 0` a `x < 0`)  
- Dot produkty: `std::fma` + Neumaierova kompenzace  
- MSE a batch MSE: kompenzované sumace  
- Učení: SGD s **gradient clippingem (±5.0)**  

---

## Omezení a doporučení

- Sekvenční provedení (bez vláken); každý handle je nezávislý  
- Nepřenášejte objekty STL přes hranici DLL (API je čisté C, bezpečné)  
- `in_len` a `out_len` musí odpovídat topologii sítě, jinak funkce vrátí `false`  
- Aktivace: `0=SIG`, `1=RELU`, `2=TANH`, `3=LINEAR`, `4=SYM_SIG`  

---

## Struktura repozitáře

```
NNMQL5FIX/
├─ src/
│  ├─ dllmain.cpp
│  ├─ pch.h
│  └─ pch.cpp
├─ build/           # výstupy (DLL, LIB, PDB)
├─ examples/
│  └─ MQL5/         # ukázky importu v indikátoru/EA
└─ README.md
```

---

## Licence

MIT-like spirit — používejte svobodně, ale prosím uveďte **Remind — Tomáš Bělák**.  
Bez záruky; software je poskytován **„tak jak je“**.

---

## Poděkování

Díky komunitě kolem MQL5 a všem, kdo mají rádi čistou numeriku a jednoduché API.  
Pull requesty vítány.
