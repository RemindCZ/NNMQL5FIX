High-Performance MLP DLL for MQL5 (x64)

A high-performance C++ DLL library implementing a Multi-Layer Perceptron (MLP) designed for MetaTrader 5 (MQL5). Focused on speed, numerical stability, and safe DLL lifecycle management.

Version: 3.2
Platform: Windows x64, MSVC, OpenMP
Author: Remind (2025)

Key Features

Dense MLP neural network

AdamW optimizer (decoupled weight decay)

OpenMP multithreaded training

High numerical precision (Neumaier summation, FMA)

Stateless layer architecture

Binary save and load of models

Built-in GDI+ real-time MSE graph

Explicit cleanup to avoid Loader Lock crashes

Activation Functions

Supported activation types (ActKind):

Code | Function
0 | Sigmoid
1 | ReLU
2 | Tanh
3 | Linear
4 | Symmetric Sigmoid (-1..1)

Exported API
Network Management

int NN_Create();
void NN_Free(int handle);
bool NN_AddDense(int handle, int in, int out, int act);

Training and Inference

bool NN_TrainBatch(
int handle,
const double* input,
int batch,
int input_len,
const double* target,
int target_len,
double lr,
double* mean_mse
);

bool NN_Forward(
int handle,
const double* input,
int input_len,
double* output,
int output_len
);

Persistence

bool NN_Save(int handle, const wchar_t* path);
bool NN_Load(int handle, const wchar_t* path);

Helpers

int NN_InputSize(int handle);
int NN_OutputSize(int handle);
void NN_SetSeed(unsigned int seed);

MSE Monitor (GDI+)

void NN_MSE_Push(double mse);
void NN_MSE_Show(int show);
void NN_MSE_Clear();
void NN_MSE_SetMaxPoints(int n);
void NN_MSE_SetAutoScale(int enable, double min, double max);

The MSE monitor runs in a separate topmost window, supports auto-scaling or fixed Y-axis ranges, and is executed outside DllMain to ensure safe multithreading.

CRITICAL: Cleanup

This function MUST be called from OnDeinit() in MQL5:

void NN_GlobalCleanup();

It safely stops all UI threads, frees all neural network instances, and prevents Loader Lock deadlocks. For safety reasons, it is intentionally not executed from DllMain.

Build Requirements

Visual Studio 2022

x64 Release configuration

/openmp enabled

Windows SDK with GDI+ support

Minimal MQL5 Usage Example

int h = NN_Create();
NN_AddDense(h, 10, 32, 1);
NN_AddDense(h, 32, 1, 3);

// training / inference ...

NN_GlobalCleanup();
