# NNMQL5FIX.dll — README (CPU + CUDA) pro MQL5 (x64)

Tento dokument krok‑za‑krokem vysvětluje **import a použití všech exportovaných funkcí** z `NNMQL5FIX.dll` v prostředí **MetaTrader 5 (MQL5, 64‑bit, cdecl)**. Pokrývá CPU MLP vrstvy, LSTM (inference i trénink), MSE monitor okno i CUDA diagnostiku/test.

> Testováno pro Windows 10/11, MT5 x64. DLL musí být zkompilována jako **x64** a umístěna do `MQL5\Libraries\NNMQL5FIX.dll`. V MT5 povolte *Nástroje → Možnosti → Experti → Povolit import DLL* (a *Důvěřovat importům* dle potřeby).


---

## 1) Rychlý přehled exportů

### CPU (MLP – husté vrstvy)
- `int   NN_Create()` / `void NN_Free(int h)`
- `bool  NN_AddDense(int h,int inSz,int outSz,int act)` – `act`: 0=SIGMOID,1=RELU,2=TANH,3=LINEAR,4=SYM_SIG
- `int   NN_InputSize(int h)` / `int NN_OutputSize(int h)`
- `bool  NN_Forward(int h,const double &in[],int in_len,double &out[],int out_len)`
- `bool  NN_TrainOne(int h,const double &in[],int in_len,const double &tgt[],int tgt_len,double lr,double &mse)`
- `bool  NN_ForwardBatch(int h,const double &in[],int batch,int in_len,double &out[],int out_len)`
- `bool  NN_TrainBatch(int h,const double &in[],int batch,int in_len,const double &tgt[],int tgt_len,double lr,double &mean_mse)`
- `bool  NN_GetWeights(int h,int i,double &W[],int Wlen,double &b[],int blen)`
- `bool  NN_SetWeights(int h,int i,const double &W[],int Wlen,const double &b[],int blen)`
- **Offline trénink na celé řadě:**  
  `void NN_TrainSeries(int h,const double *X,int x_len,const double *Y,int y_len,int win_in,int win_out,int lead,int epochs,double lr,double target_mse,int batch_size,int shuffle)`

### LSTM (více vrstev, SGD, TBPTT, clipping)
- Životní cyklus/info:  
  `int LSTM_Create(int in_sz,int hid_sz,int out_sz,int num_layers)` / `void LSTM_Free(int h)` /  
  `int LSTM_Reset(int h)` / `int LSTM_GetInfo(int h,int &in_sz,int &hid_sz,int &out_sz,int &num_layers)`
- Inference:  
  `bool LSTM_ForwardLast(int h,const double &seq[],int seq_len,double &out[],int out_len)`  
  `bool LSTM_ForwardSeq (int h,const double &seq[],int seq_len,double &out[],int out_len)`
- Trénink (online/batch/offline série):  
  `bool LSTM_TrainOne(int h,const double &seq[],int seq_len,const double &tgt[],int tgt_len,double lr,double &mse)`  
  `bool LSTM_TrainBatch(int h,const double &seq_batch[],int batch,int seq_len,const double &tgt_batch[],int tgt_len,double lr,double &mean_mse)`  
  `void LSTM_TrainSeries(int h,const double *X,int x_len,const double *Y,int y_len,int win_in,int win_out,int lead,int epochs,double lr,double target_mse,int batch_size,int tbptt_k,int shuffle)`
- Nastavení tréninku & váhy:  
  `int  LSTM_SetTBPTT(int h,int k_steps)` / `int  LSTM_SetClipGrad(int h,double clip_norm)`  
  `bool LSTM_SetOutWeights(...)` / `bool LSTM_GetOutWeights(...)`  
  `bool LSTM_SetLayerWeights(h,layer_index,Wx,Wxlen,Wh,Whlen,b,blen)` / `bool LSTM_GetLayerWeights(...)`

### MSE monitor okno
- `void NN_MSE_Show(int show)` / `void NN_MSE_Push(double mse)` / `void NN_MSE_Clear()`  
- `void NN_MSE_SetMaxPoints(int n)` / `void NN_MSE_SetAutoScale(int enable,double y_min,double y_max)`

### Chyby & cancel
- `int  NN_GetLastError(int h,int &code,ushort &buf[],int buf_len)` / `void NN_SetCancel(int h,int flag)`  
- `int  LSTM_GetLastError(int h,int &code,ushort &buf[],int buf_len)` / `void LSTM_SetCancel(int h,int flag)`

> Pozn.: `buf` přijímá **UTF‑16 (wchar_t)**. V MQL5 tedy používejte **`ushort` buffer** a `ShortArrayToString(...)`.

### CUDA diagnostika + test
- `int  NN_CUDA_Available()` – alespoň 1 GPU
- `int  NN_CUDA_RuntimeVersion()` / `int NN_CUDA_DriverVersion()`
- `int  NN_CUDA_TestAdd(const double &a[],const double &b[],double &out[],int n)` – vektorové sčítání na GPU
- `const char* NN_GetLastCuda()` – poslední CUDA chyba jako **ANSI C‑řetězec** (čtěte přes `uchar` buffer)

---

## 2) Importy v MQL5

```mql5
// NNMQL5FIX.dll – cdecl, x64
#import "NNMQL5FIX.dll"

// --- CPU / MLP ---
int  NN_Create();
void NN_Free(int h);

bool NN_AddDense(int h,int inSz,int outSz,int act);
int  NN_InputSize(int h);
int  NN_OutputSize(int h);

bool NN_Forward(int h,const double &in[],int in_len,double &out[],int out_len);
bool NN_TrainOne(int h,const double &in[],int in_len,const double &tgt[],int tgt_len,double lr,double &mse);
bool NN_ForwardBatch(int h,const double &in[],int batch,int in_len,double &out[],int out_len);
bool NN_TrainBatch(int h,const double &in[],int batch,int in_len,const double &tgt[],int tgt_len,double lr,double &mean_mse);

bool NN_GetWeights(int h,int i,double &W[],int Wlen,double &b[],int blen);
bool NN_SetWeights(int h,int i,const double &W[],int Wlen,const double &b[],int blen);

void NN_TrainSeries(int h,const double *X,int x_len,const double *Y,int y_len,
                    int win_in,int win_out,int lead,int epochs,double lr,double target_mse,
                    int batch_size,int shuffle);

// --- LSTM ---
int  LSTM_Create(int in_sz,int hid_sz,int out_sz,int num_layers);
void LSTM_Free(int h);
int  LSTM_Reset(int h);
int  LSTM_GetInfo(int h,int &in_sz,int &hid_sz,int &out_sz,int &num_layers);

bool LSTM_ForwardLast(int h,const double &seq[],int seq_len,double &out[],int out_len);
bool LSTM_ForwardSeq (int h,const double &seq[],int seq_len,double &out[],int out_len);

bool LSTM_TrainOne  (int h,const double &seq[],int seq_len,const double &tgt[],int tgt_len,double lr,double &mse);
bool LSTM_TrainBatch(int h,const double &seq_batch[],int batch,int seq_len,const double &tgt_batch[],int tgt_len,double lr,double &mean_mse);
void LSTM_TrainSeries(int h,const double *X,int x_len,const double *Y,int y_len,
                      int win_in,int win_out,int lead,int epochs,double lr,double target_mse,
                      int batch_size,int tbptt_k,int shuffle);

int  LSTM_SetTBPTT(int h,int k_steps);
int  LSTM_SetClipGrad(int h,double clip_norm);

bool LSTM_SetOutWeights(int h,const double &W[],int Wlen,const double &b[],int blen);
bool LSTM_GetOutWeights(int h,double &W[],int Wlen,double &b[],int blen);

bool LSTM_SetLayerWeights(int h,int layer_index,const double &Wx[],int Wxlen,
                          const double &Wh[],int Whlen,const double &b[],int blen);
bool LSTM_GetLayerWeights(int h,int layer_index,double &Wx[],int Wxlen,
                          double &Wh[],int Whlen,double &b[],int blen);

// --- MSE monitor okno ---
void NN_MSE_Show(int show);
void NN_MSE_Push(double mse);
void NN_MSE_Clear();
void NN_MSE_SetMaxPoints(int n);
void NN_MSE_SetAutoScale(int enable,double y_min,double y_max);

// --- Error / Cancel (UTF‑16 buffer!) ---
int  NN_GetLastError  (int h,int &code,ushort &buf[],int buf_len);
void NN_SetCancel     (int h,int flag);
int  LSTM_GetLastError(int h,int &code,ushort &buf[],int buf_len);
void LSTM_SetCancel   (int h,int flag);

// --- CUDA ---
int  NN_CUDA_Available();
int  NN_CUDA_RuntimeVersion();
int  NN_CUDA_DriverVersion();
int  NN_CUDA_TestAdd(const double &a[],const double &b[],double &out[],int n);

// ANSI char* → číst přes uchar[]
int  NN_GetLastCudaStrLen(); // (není k dispozici; použijte fixní buffer a/nebo test n = 256)
const char* NN_GetLastCuda(); // čtěte do uchar[] a konvertujte

#import
```

**Poznámky k typům:**
- `const double &arr[]` v MQL5 znamená *vstupní pole*; `double &arr[]` je *výstupní pole*.
- Pro `wchar_t*` (UTF‑16) používejte `ushort buf[]`. Pro `char*` (ANSI) používejte `uchar buf[]` a `CharArrayToString`.

---

## 3) Helper funkce (MQL5) pro čtení chyb

```mql5
string NN_ReadLastError(int h,int is_lstm=false)
{
   int code=0;
   ushort buf[]; ArrayResize(buf,1024); // UTF-16 buffer
   bool ok=false;
   if(!is_lstm)
      ok = (NN_GetLastError(h,code,buf,ArraySize(buf))==1);
   else
      ok = (LSTM_GetLastError(h,code,buf,ArraySize(buf))==1);
   if(!ok) return "no-error";
   string msg = ShortArrayToString(buf);
   return StringFormat("#%d: %s",code,msg);
}

string CUDA_ReadLastError()
{
   uchar buf[]; ArrayResize(buf,256);
   // Pozn.: NN_GetLastCuda() vrací ukazatel; v MQL5 použijte CopyMemory hack:
   // Jednodušší: zavolejte NN_GetLastCuda(); pak ručně převeďte přes kernel32 lstrcpynA
   // Pro praxi je nejjednodušší číst pouze návratové kódy a runtime/driver verze.
   return "(viz NN_GetLastCuda; doporučeno používat return kódy funkcí)";
}
```

> MQL5 nemá přímé API pro kopírování `char*` do `string`. Pro jednoduchost sledujte návraty funkcí CUDA; text z `NN_GetLastCuda()` používejte jen jako doplňkový debug (lze řešit přes WinAPI `lstrcpynA` a `kernel32`, není zahrnuto).

---

## 4) Příklady použití (MQL5)

### 4.1 MLP – minimální end‑to‑end (inference + train)

```mql5
void OnStart()
{
   int h = NN_Create();
   if(h==0){ Print("NN_Create fail"); return; }
   // topologie: 4 -> 8 -> 1 (tanh, sigmoid)
   bool ok=true;
   ok&=NN_AddDense(h,4,8,2); // TANH
   ok&=NN_AddDense(h,8,1,0); // SIGMOID
   if(!ok){ Print("AddDense fail: ",NN_ReadLastError(h)); NN_Free(h); return; }

   double x[4] = {0.1,0.2,0.3,0.4};
   double y[1]; ArrayInitialize(y,0.0);
   if(!NN_Forward(h,x,4,y,1)){ Print("Forward fail"); }
   Print("y0=",DoubleToString(y[0],6));

   // TrainOne na fiktivní target 0.75
   double tgt[1]={0.75};
   double mse=0.0;
   if(!NN_TrainOne(h,x,4,tgt,1,0.01,mse)){
      Print("TrainOne fail: ",NN_ReadLastError(h));
   }
   Print("mse=",DoubleToString(mse,8));
   NN_Free(h);
}
```

### 4.2 MLP – offline trénink na časové řadě

```mql5
void TrainSeriesMLP()
{
   const int N=1000;
   static double X[],Y[];
   ArrayResize(X,N); ArrayResize(Y,N);
   for(int i=0;i<N;i++){ double t=i*0.01; X[i]=MathSin(t); Y[i]=MathSin(t+0.2); }

   int h=NN_Create();
   NN_AddDense(h,32,32,2); // TANH
   NN_AddDense(h,32,8,2);
   NN_AddDense(h,8, 4,3);  // LINEAR
   // okna
   int win_in=32, win_out=4, lead=0;

   NN_MSE_Show(1); // zobraz MSE okno
   NN_TrainSeries(h,X,N,Y,N,win_in,win_out,lead,/*epochs*/10,/*lr*/0.005,/*target*/0.0,
                  /*batch*/32,/*shuffle*/1);

   Print("Last error: ",NN_ReadLastError(h));
   NN_Free(h);
}
```

### 4.3 LSTM – inference nad sekvencí

```mql5
void LstmForward()
{
   int in_sz=3, hid_sz=16, out_sz=2, layers=2;
   int h = LSTM_Create(in_sz,hid_sz,out_sz,layers);
   if(h==0){ Print("LSTM_Create fail"); return; }

   // sekvence T=5, in_sz=3 => pole délky 5*3
   double seq[] = { 0.1,0.2,0.3,  0.2,0.3,0.1,  0.0,0.1,0.2,  0.3,0.1,0.0,  0.2,0.2,0.2 };
   double out_last[2];
   if(!LSTM_ForwardLast(h,seq,5,out_last,2)){
      Print("LSTM_ForwardLast fail: ",NN_ReadLastError(h,true));
   } else {
      Print("y_last = ",DoubleToString(out_last[0],6),", ",DoubleToString(out_last[1],6));
   }
   LSTM_Free(h);
}
```

### 4.4 LSTM – batch trénink s TBPTT a clippingem

```mql5
void LstmTrainBatch()
{
   int h=LSTM_Create(1,32,1,1); // 1 feature, H=32, 1 output, 1 vrstva
   LSTM_SetTBPTT(h,16);         // TBPTT 16 kroků
   LSTM_SetClipGrad(h,5.0);     // L2 clip norm

   const int B=8, T=32; // batch 8, délka sekvence 32
   double seq_batch[]; ArrayResize(seq_batch,B*T*1);
   double tgt_batch[]; ArrayResize(tgt_batch,B*1);

   // dummy data: s(t)=sin, cíl = s(t+1)
   for(int b=0;b<B;b++){
      for(int t=0;t<T;t++){
         double s=MathSin((b*T+t)*0.05);
         seq_batch[b*T*1 + t*1 + 0] = s;
      }
      tgt_batch[b*1 + 0] = MathSin((b*T+T)*0.05);
   }

   NN_MSE_Show(1);
   for(int e=0;e<20;e++){
      double mean_mse=0.0;
      bool ok=LSTM_TrainBatch(h,seq_batch,B,T,tgt_batch,1,0.01,mean_mse);
      if(!ok){ Print("TrainBatch fail: ",NN_ReadLastError(h,true)); break; }
      PrintFormat("[E%02d] MSE=%.6f",e,mean_mse);
   }
   LSTM_Free(h);
}
```

### 4.5 CUDA – diagnostika a test kernel

```mql5
void CudaTest()
{
   if(NN_CUDA_Available()!=1){
      Print("CUDA not available. Driver=",NN_CUDA_DriverVersion()," Runtime=",NN_CUDA_RuntimeVersion());
      return;
   }
   const int n=1024;
   double a[],b[],c[];
   ArrayResize(a,n); ArrayResize(b,n); ArrayResize(c,n);
   for(int i=0;i<n;i++){ a[i]=i; b[i]=2*i; }
   int ok = NN_CUDA_TestAdd(a,b,c,n);
   if(ok!=1){
      Print("NN_CUDA_TestAdd failed");
      return;
   }
   PrintFormat("c[0]=%.1f c[n-1]=%.1f",c[0],c[n-1]); // oček. 0, 3*(n-1)
}
```

---

## 5) Semantika vybraných funkcí

### 5.1 `NN_TrainSeries` (MLP)
- Vytváří **slide‑window** vstupy z `X` (délka `x_len`) velikosti `win_in` a cíle z `Y` posunuté o `lead` o délce `win_out`.
- Počet vzorků: `N = min(x_len-(win_in+lead+win_out)+1, y_len-(lead+win_out))`. Pokud `N<=0` → chyba.
- Uvnitř volá opakovaně `NN_TrainBatch` nad promíchanými indexy (`shuffle` ∈ {0,1}).
- `target_mse > 0` → **časné ukončení**, jakmile epoch MSE ≤ target.
- `NN_SetCancel(h,1)` umožní **předčasné ukončení** během epoch.

### 5.2 `LSTM_TrainSeries`
- Stejná logika posuvného okna jako u MLP, ale sekvenční vstup má tvar `[win_in × in_sz]` (feature vektory). Pokud `in_sz==1`, očekává se 1D řada.
- Parametr `tbptt_k` (0 = plný BPTT) omezuje backprop na posledních `k` časových kroků.
- `LSTM_SetClipGrad(h, norm)` zapíná **globální L2 gradient clipping** přes všechny parametry.
- Uvnitř používá `LSTM_TrainBatch` a průběžně posílá MSE do monitoru.

### 5.3 MSE monitor okno
- Samostatné top‑level okno (ALWAYS‑ON‑TOP). `NN_MSE_Show(1)` spustí vlákno GUI a zobrazí okno.
- Data přidávejte přes `NN_MSE_Push(mse)`; automaticky se překreslí.
- `NN_MSE_SetAutoScale(0,ymin,ymax)` nastaví fixní vertikální rozsah. `NN_MSE_SetMaxPoints(n)` limituje historii.

---

## 6) Best‑practices (MQL5, výkon, stabilita)

- **64‑bit**: meta‑terminál i DLL musí být x64; v opačném případě chyba `193` (nesprávný formát).
- **cdecl**: exporty jsou `__cdecl` (MQL5 default), není třeba `#pragma`.
- **Paměť polí**: MQL5 alokuje/rozšiřuje přes `ArrayResize`; vždy zajistěte správné délky (`in_len/out_len`).
- **Normalizace**: trénink předpokládá rozumné měřítko, ideálně v rozsahu `[-1,1]` nebo `[0,1]`.
- **Ukončení**: u dlouhých běhů `NN_TrainSeries`/`LSTM_TrainSeries` mějte UI klávesu pro `*_SetCancel(h,1)`.
- **Diagnostika**: při selháních čtěte `NN_GetLastError/LSTM_GetLastError`; pro CUDA sledujte dostupnost a verze.
- **Okna**: MSE okno je „always‑on‑top“; pokud překáží, skryjte `NN_MSE_Show(0)`.

---

## 7) Mapování aktivací (MLP)

| `act` | Název        | Poznámka                         |
|------:|--------------|----------------------------------|
| 0     | SIGMOID      | stabilní implementace            |
| 1     | RELU         | He init                          |
| 2     | TANH         |                                  |
| 3     | LINEAR       | bez nelinearity                  |
| 4     | SYM_SIG      | `2*sigmoid(x)-1`                 |

---

## 8) Řešení problémů

- **„failed to load“ / 126/193**: zkontrolujte x64 build, umístění do `MQL5\Libraries`, závislosti (VC Runtime, CUDA).
- **CUDA not available**: ověřte driver (Studio/GRD), `NN_CUDA_DriverVersion()` a `NN_CUDA_RuntimeVersion()`.
- **„Topology mismatch“**: velikosti `in_len/out_len` musí odpovídat poslední vrstvě (MLP) a `win_in/win_out` u TrainSeries.
- **„Series too short for given windows“**: upravte `win_in/lead/win_out` nebo prodlužte data.
- **Mrznoucí trénink**: použijte menší `batch_size`, ověřte `shuffle`, případně `SetCancel` a nižší `lr`.

---

## 9) Licenční poznámka

„MIT‑like spirit — use freely, please keep attribution.“
Zachovejte informaci o autorství při dalším šíření.

---

## 10) Rychlé šablony (kopírovatelný start)

```mql5
// --- GPU ping ---
void OnStart(){
   Print("[CUDA] avail=",NN_CUDA_Available()," drv=",NN_CUDA_DriverVersion()," rt=",NN_CUDA_RuntimeVersion());
}
```

```mql5
// --- MSE monitor toggle ---
void ToggleMse(bool show=true){
   NN_MSE_Show(show?1:0);
   if(show){
      NN_MSE_Clear();
      NN_MSE_SetAutoScale(1,0,0);
      NN_MSE_SetMaxPoints(1000);
   }
}
```
