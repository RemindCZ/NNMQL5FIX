# Repository Guidelines

## Project Structure & Module Organization
- Root holds `NNMQL5FIX.sln`, `README.md`, and licensing; all C++ sources live in `NNMQL5FIX/`.
- `dllmain.cpp` exposes the exported MLP/LSTM API; `pch.*` and `framework.h` support the MSVC build.
- Built artifacts land in `x64/Debug` and `x64/Release`; treat them as disposable outputs and avoid manual edits.
- Intermediate objects also appear under `NNMQL5FIX/x64`; clean these via Visual Studio before commits.

## Build, Test, and Development Commands
- `msbuild NNMQL5FIX.sln /p:Configuration=Debug /p:Platform=x64` builds a debug DLL with `/MTd`.
- `msbuild NNMQL5FIX.sln /p:Configuration=Release /p:Platform=x64` produces the shipping DLL in `x64\Release`.
- `dumpbin /dependents x64\Release\NNMQL5FIX.dll` verifies the binary stays free of VC++ runtime deps.
- When iterating through the IDE, ensure the project still references `pch.cpp` as **Create** and other `.cpp` files as **Use**.

## Coding Style & Naming Conventions
- Target C++17, four-space indentation, and brace-on-same-line for functions, namespaces, and structs.
- Always include `pch.h` first in every translation unit; keep includes ordered from project headers to STL.
- Exported entry points follow `NN_*` or `LSTM_*` uppercase naming; helper classes prefer `CamelCase`, local variables `snake_case`.
- Favor deterministic numerics: stick with `std::fma`, Neumaier-style reducers, and explicit casts already used in `dllmain.cpp`.

## Testing Guidelines
- No automated suite exists; rely on MetaTrader 5 scripts that import `NNMQL5FIX.dll` for behavioral checks.
- Re-run the README mini-batch example after structural changes to confirm gradients and MSE outputs stay stable.
- Check both Debug and Release builds for consistent loss trends and absence of warnings in the Visual Studio output window.
- Capture regression evidence (logs or CSV traces) when adjusting training loops or accumulation logic.

## Commit & Pull Request Guidelines
- Keep commits scoped; use short imperative subjects (`Fix TBPTT window clamp`) consistent with the existing history.
- Summaries should call out numerical or API impact and list any updated docs or MetaTrader scripts.
- Pull requests describe build configs exercised, include reproduction steps, and link related issues or tasks.
- Attach `dumpbin` output or screenshots when DLL exports or dependencies change to speed manual verification.
