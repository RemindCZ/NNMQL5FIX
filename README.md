# High-Performance MLP DLL for MQL5 (x64)

**Version:** 3.2
**Author:** Remind — Optimized Architecture
**Date:** 2025

This repository contains a high-performance C++ Dynamic Link Library (DLL) implementing a Multi-Layer Perceptron (MLP) neural network. It is specifically designed for integration with **MetaTrader 5 (MQL5)**. The library utilizes OpenMP for parallel processing, AVX/FMA instructions for mathematical precision, and the AdamW optimizer for efficient convergence.

## Key Features

*   **AdamW Optimization:** Implements the AdamW algorithm with decoupled weight decay for superior generalization compared to standard SGD or Adam.
*   **OpenMP Parallelism:** Multi-threaded batch training capable of utilizing all available CPU cores.
*   **High Precision Math:** Uses FMA (Fused Multiply-Add) and Neumaier summation to minimize floating-point accumulation errors during gradient descent.
*   **Stateless Architecture:** Layers are designed to be memory efficient.
*   **Binary Persistence:** Fast, compact binary format for saving and loading models.
*   **Built-in Visualization:** Includes a lightweight GDI+ window to graph Mean Squared Error (MSE) in real-time.
*   **Loader Lock Safety:** Version 3.2 addresses Windows Loader Lock deadlocks by implementing an explicit cleanup mechanism (`NN_GlobalCleanup`) instead of relying on `DllMain` detachment logic.

## Prerequisites & Compilation

To build this project, you need a C++ compiler compatible with MSVC (Visual Studio 2019 or newer is recommended).

### Project Settings
Ensure your project is configured with the following settings:

1.  **Architecture:** x64 (Required for MQL5).
2.  **Language Standard:** C++17 or later.
3.  **OpenMP Support:** Enabled (`/openmp`). This is critical for performance.
4.  **Floating Point Model:** Precise or Strict (recommended for Neumaier summation consistency).
5.  **Runtime Library:** Multi-threaded DLL (`/MD` or `/MDd`).

## API Reference

All functions are exported using `extern "C"` and the `__cdecl` calling convention.

### Lifecycle Management
*   `int NN_Create()`
    Allocates a new neural network instance. Returns a unique integer handle.
*   `void NN_Free(int handle)`
    Frees a specific network instance.
*   `void NN_GlobalCleanup()`
    **CRITICAL:** Must be called once before the MQL5 script/EA terminates. It safely shuts down the UI thread and releases all resources to prevent deadlocks.

### Topology Configuration
*   `bool NN_AddDense(int handle, int input_size, int output_size, int activation)`
    Adds a fully connected layer.
    *   **Activation Codes:** 0=Sigmoid, 1=ReLU, 2=Tanh, 3=Linear, 4=Symmetric Sigmoid.
*   `void NN_SetSeed(unsigned int seed)`
    Sets the RNG seed for weight initialization and shuffling.

### Training & Inference
*   `bool NN_Forward(int handle, const double* input, int in_len, double* output, int out_len)`
    Computes the forward pass for a single input vector.
*   `bool NN_TrainBatch(int handle, const double* inputs, int batch_size, int input_len, const double* targets, int target_len, double lr, double* out_mse)`
    Performs a backward pass (training) on a batch of data using OpenMP.
    *   `inputs`: Flat array containing `batch_size * input_len` elements.
    *   `out_mse`: Optional pointer to receive the mean squared error of the batch.

### Persistence
*   `bool NN_Save(int handle, const wchar_t* filename)`
    Saves the model weights and topology to a binary file.
*   `bool NN_Load(int handle, const wchar_t* filename)`
    Loads a model from a binary file.

### Visualization (MSE Monitor)
*   `void NN_MSE_Show(int show)`
    Show (1) or hide (0) the GDI+ graphing window.
*   `void NN_MSE_Push(double mse)`
    Push a new error value to the graph.
*   `void NN_MSE_Clear()`
    Clears the graph data.
*   `void NN_MSE_SetMaxPoints(int points)`
    Sets the rolling window size for the graph.
*   `void NN_MSE_SetAutoScale(int enable, double min, double max)`
    Enables autoscaling or sets fixed Y-axis bounds.

## MQL5 Integration Example

Below is a snippet demonstrating how to properly import and manage the DLL within an Expert Advisor (EA).

```cpp
// MLP_Bridge.mqh

#import "HighPerfMLP.dll"
   int  NN_Create();
   void NN_Free(int h);
   bool NN_AddDense(int h, int i, int o, int act);
   bool NN_TrainBatch(int h, const double &in[], int batch, int il, 
                      const double &tgt[], int tl, double lr, double &mse);
   bool NN_Forward(int h, const double &in[], int il, double &out[], int ol);
   
   // Critical cleanup function
   void NN_GlobalCleanup();
   
   // UI Functions
   void NN_MSE_Show(int s);
   void NN_MSE_Push(double m);
#import

// --- Expert Advisor Example ---

int net_handle = -1;

int OnInit() {
   // Initialize Network
   net_handle = NN_Create();
   if (net_handle <= 0) return INIT_FAILED;

   // Define Architecture: 10 Inputs -> 64 Hidden (ReLU) -> 1 Output (Sigmoid)
   NN_AddDense(net_handle, 10, 64, 1); // 1 = ReLU
   NN_AddDense(net_handle, 64, 1, 0);  // 0 = Sigmoid
   
   // Show Training Monitor
   NN_MSE_Show(1);
   
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   // CRITICAL: Call GlobalCleanup to stop threads safely
   // Do not rely on MQL5 to free the library automatically
   NN_GlobalCleanup();
}

void OnTick() {
   if (net_handle <= 0) return;

   // Example: Prepare dummy data (Replace with real market data)
   double inputs[10];
   double target[1];
   ArrayInitialize(inputs, 0.5);
   target[0] = 1.0;
   
   double mse = 0.0;
   double learning_rate = 0.001;
   
   // Train on a single sample (Batch size = 1)
   if (NN_TrainBatch(net_handle, inputs, 1, 10, target, 1, learning_rate, mse)) {
      NN_MSE_Push(mse);
   }
}
