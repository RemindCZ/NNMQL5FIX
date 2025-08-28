# NNMQL5FIX
**NNMQL5FIX — Lehká MLP DLL pro MetaTrader 5 (x64, MSVC)**

Autor: **Remind — Tomáš Bělák**  
Licence: **MIT-like** (užijte, ale uveďte autorství)

Malá, numericky stabilní MLP knihovna jako DLL pro MQL5.  
Bez externích závislostí, čisté C API, 64-bit, MSVC.  
Funkce: dopředný průchod, učení (SGD), mini-batch obálky, přístup k vahám.

---

## Vlastnosti
- Stabilní sigmoid (bez overflow/underflow).  
- Přesné dot-produkty: Neumaierova kompenzace + FMA.  
- Kompenzované sumace MSE i batch MSE.  
- Gradient clipping a konzervativní numerika.  
- Čisté C exporty (žádné STL přes hranici DLL).  
- Build s `/MT` (statický CRT) → bez VC++ Redistributable.  

**Pozn.:** Tato verze je sekvenční (paralelismus vypnutý).

---

## Build (Visual Studio 2022, x64)

### PCH (precompiled headers)
**pch.h (v projektu):**
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
