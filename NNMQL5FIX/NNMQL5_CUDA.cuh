#pragma once

#ifndef DLL_EXTERN
#define DLL_EXTERN extern "C" __declspec(dllexport)
#endif
#ifndef DLL_CALL
#define DLL_CALL __cdecl
#endif

// Všechny funkce exportované z CUDA modulu
DLL_EXTERN const char* DLL_CALL NN_GetLastCuda();
DLL_EXTERN int         DLL_CALL NN_CUDA_Available();
DLL_EXTERN int         DLL_CALL NN_CUDA_RuntimeVersion();
DLL_EXTERN int         DLL_CALL NN_CUDA_DriverVersion();
DLL_EXTERN int         DLL_CALL NN_CUDA_TestAdd(const double* a, const double* b, double* out, int n);
