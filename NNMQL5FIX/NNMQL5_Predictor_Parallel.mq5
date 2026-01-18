//+------------------------------------------------------------------+
//|                       NNMQL5_Predictor_BinSaveLoad.mq5           |
//| V3.1: Binary Save/Load + OpenMP AdamW                            |
//| Vyžaduje: NNMQL5FIX.dll (C++ Ver 3.1)                            |
//+------------------------------------------------------------------+
#property copyright "Remind"
#property link      "https://remind.cz"
#property version   "3.10"
#property description "MLP Predictor s podporou binárního ukládání a GDI+ grafu chyby."

#property strict
#property indicator_chart_window
#property indicator_buffers 2
#property indicator_plots   2

// --- VIZUALIZACE ---
#property indicator_label1  "Trend Pred +1"
#property indicator_type1   DRAW_LINE
#property indicator_style1  STYLE_SOLID
#property indicator_width1  2
#property indicator_color1  clrDodgerBlue

#property indicator_label2  "Target +1"
#property indicator_type2   DRAW_LINE
#property indicator_style2  STYLE_DOT
#property indicator_width2  1
#property indicator_color2  clrSilver

// --- DLL IMPORT ---
// Ujistěte se, že soubor se jmenuje přesně takto a je v MQL5/Libraries
#import "NNMQL5FIX.dll"
   int  NN_Create();
   void NN_Free(int h);
   bool NN_AddDense(int h, int i, int o, int a);
   // Všimněte si 'double &in[]', což předává ukazatel na pole (C++: double*)
   bool NN_TrainBatch(int h, double &in[], int b, int il, double &t[], int tl, double lr, double &mse);
   bool NN_Forward(int h, double &in[], int il, double &out[], int ol);
   
   // Save/Load bere string (MQL5 string je unicode -> wchar_t*)
   bool NN_Save(int h, string p);
   bool NN_Load(int h, string p);
   
   int  NN_InputSize(int h);
   int  NN_OutputSize(int h);
   void NN_SetSeed(uint s);
   
   // UI Funkce
   void NN_MSE_Push(double m);
   void NN_MSE_Show(int s);
   void NN_MSE_Clear();
   void NN_MSE_SetMaxPoints(int n);
   void NN_MSE_SetAutoScale(int e, double mn, double mx);
#import

// --- VSTUPNÍ PARAMETRY ---
input group "Strategy & Data"
input bool     UseLogReturns  = true;      // TRUE = Učí se změny ceny (Log-returns)
input int      Lookback       = 64;        // Vstupní okno (počet svíček zpět)
input int      ForecastHorizon= 10;        // Výhled predikce (počet kroků dopředu)
input ENUM_APPLIED_PRICE PriceMode = PRICE_CLOSE;

input group "Network Architecture"
input int      Hidden         = 64;        // Velikost skryté vrstvy
input int      Depth          = 3;         // Počet skrytých vrstev
// 0=Sigmoid, 1=ReLU, 2=Tanh, 3=Linear
input int      HiddenAct      = 2;         // 2=TANH (doporučeno pro normalizovaná data)
input double   LR             = 0.001;     // Rychlost učení (AdamW)

input group "Training Control"
input int      TrainBars      = 2000;      // Kolik svíček historie použít pro trénink
input double   TargetMSE      = 0.000005;  // Cílová chyba pro zastavení
input double   MaxMSEGuard    = 10.0;      // Pojistka proti explozi gradientu
input bool     UseTimer       = true;      // Automaticky spustit trénink při startu
input bool     DebugLog       = true;      // Výpis do logu

input group "File Operations"
input string   FileName       = "MyNet.bin"; // Název souboru (bude v MQL5/Files)

input group "Visualization"
input bool     ShowFuture     = true;      // Kreslit predikci do budoucna
input int      FuturePts      = 30;        // Délka budoucí křivky
input color    ColorUp        = clrLimeGreen;
input color    ColorDown      = clrTomato;
input color    ColorFlat      = clrSilver;
input int      WidthFuture    = 2;

// --- BUFFERS ---
double PredBuffer[];
double TgtBuffer[];

// --- GLOBALS ---
int    g_h = 0;              // Handle sítě
int    g_epochs = 0;
bool   g_ready = false;      // Je síť připravena k predikci?
double g_mean = 0.0;         // Průměr pro normalizaci
double g_std = 1.0;          // Směrodatná odchylka
double g_X[];                // Trénovací vstupy (Flattened)
double g_T[];                // Trénovací cíle (Flattened)
int    g_N = 0;              // Počet trénovacích vzorků
bool   g_mse_visible = false; 
bool   g_timer_running = false; 

// --- GUI NAMES ---
string g_btn_mse  = "Btn_ToggleMSE";
string g_btn_save = "Btn_SaveBin"; 
string g_btn_load = "Btn_LoadBin";

// --- PROTOTYPES ---
void CreateGUI();
void RecreateNet();
bool BuildDataset(const double &o[], const double &h[], const double &l[], const double &c[], int total);
void CalculateHistory(const double &o[], const double &h[], const double &l[], const double &c[], int total);
void DrawFuture(const double &o[], const double &h[], const double &l[], const double &c[], int total, const datetime &tm[]);
void ForceRepaint();
double GetVal(int i, const double &o[], const double &h[], const double &l[], const double &c[]);
void StartTimerSafe(); 
void StopTimer();
void SaveModelBin(); 
void LoadModelBin();

//+------------------------------------------------------------------+
//| INIT                                                             |
//+------------------------------------------------------------------+
int OnInit()
{
   NN_SetSeed(GetTickCount()); // Random seed v C++
   
   SetIndexBuffer(0, PredBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, TgtBuffer,  INDICATOR_DATA);
   
   PlotIndexSetInteger(0, PLOT_SHIFT, 1);
   PlotIndexSetInteger(1, PLOT_SHIFT, 1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);

   // Vytvoření tlačítek
   CreateGUI();
   
   // Inicializace sítě
   RecreateNet();

   // Nastavení externího grafu
   NN_MSE_SetMaxPoints(1000);
   NN_MSE_SetAutoScale(1, 0, 1);
   NN_MSE_Show(0); // Ve výchozím stavu skryté
   
   if(UseTimer) StartTimerSafe();
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| DEINIT                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   
   // Odstranění GUI
   ObjectDelete(0, g_btn_mse);
   ObjectDelete(0, g_btn_save);
   ObjectDelete(0, g_btn_load);
   ObjectsDeleteAll(0, "NNFUT_");
   
   // Zavření externího okna a uvolnění DLL
   NN_MSE_Show(0);
   NN_Free(g_h);
}

//+------------------------------------------------------------------+
//| CALCULATION                                                      |
//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
{
   int need = TrainBars + Lookback + ForecastHorizon + 1;
   if(rates_total < need) return 0;
   
   // Pokud přibyla nová svíčka nebo je to první spuštění
   if(rates_total != prev_calculated) 
   {
      ArrayInitialize(PredBuffer, EMPTY_VALUE);
      ArrayInitialize(TgtBuffer, EMPTY_VALUE);
      
      // 1. Příprava dat (Normalizace)
      if(BuildDataset(open, high, low, close, rates_total))
      {
         // Pokud běží časovač, OK. Pokud neběží a síť není ready, zkusíme ji nakopnout.
         if(!g_ready && UseTimer && !g_timer_running) {
             StartTimerSafe();
         }

         // Pokud se právě trénuje, provedeme jeden krok i zde (pro rychlejší odezvu)
         if(g_timer_running && g_N > 0) {
             double mse=0;
             // Celý dataset jako jeden Batch (díky OpenMP v DLL je to efektivní)
             NN_TrainBatch(g_h, g_X, g_N, Lookback, g_T, 1, LR, mse);
         }
         
         // 2. Inference (Výpočet predikcí)
         // Pouze pokud je síť natrénovaná nebo načtená
         if(g_ready) {
             CalculateHistory(open, high, low, close, rates_total);
             DrawFuture(open, high, low, close, rates_total, time);
         }
      }
   }
   
   return rates_total;
}

//+------------------------------------------------------------------+
//| TIMER - Učení (Backpropagation)                                  |
//+------------------------------------------------------------------+
void OnTimer()
{
   if(g_h <= 0 || g_N <= 0) return;
   
   double mse = 0.0;
   // Trénujeme na celém datasetu g_N vzorků najednou
   // DLL si to interně rozdělí mezi vlákna CPU
   bool ok = NN_TrainBatch(g_h, g_X, g_N, Lookback, g_T, 1, LR, mse);
   
   g_ready = true; // Síť už něco umí
   
   if(ok)
   {
      // Ochrana proti explozi
      if(mse > MaxMSEGuard || !MathIsValidNumber(mse)) {
         PrintFormat("CRITICAL: MSE Explosion (%.5f). Resetting network...", mse);
         RecreateNet();
         g_ready = false;
         return;
      }
      
      // Update grafu
      NN_MSE_Push(mse);
      g_epochs++;
      
      // Logování
      if(DebugLog && (g_epochs % 50 == 0)) {
         PrintFormat("Train Epoch: %d | MSE: %.8f", g_epochs, mse);
      }

      // Podmínka stopu
      if(mse < TargetMSE) {
         if(DebugLog) PrintFormat("SUCCESS: Target MSE %.8f reached. Training paused.", TargetMSE);
         StopTimer();
         ForceRepaint();
         return;
      }
      
      // Překreslit graf každých 10 epoch, abychom viděli progress vizuálně
      if(g_epochs % 10 == 0) {
         ForceRepaint();
      }
   }
}

//+------------------------------------------------------------------+
//| ULOŽENÍ BINÁRNÍ SÍTĚ                                             |
//+------------------------------------------------------------------+
void SaveModelBin()
{
   if(g_h <= 0) {
      Print("Warning: Network not valid.");
      return;
   }
   
   // Cesta do MQL5/Files
   string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + FileName;
   
   if(NN_Save(g_h, path)) {
      Print("OK: Network saved to: ", path);
      PlaySound("Ok.wav");
   } else {
      Print("ERROR: Failed to save network to: ", path);
      PlaySound("Timeout.wav");
   }
}

//+------------------------------------------------------------------+
//| NAČTENÍ BINÁRNÍ SÍTĚ                                             |
//+------------------------------------------------------------------+
void LoadModelBin()
{
   if(g_h <= 0) RecreateNet(); 

   string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + FileName;
   
   // Zastavit trénink před načtením
   StopTimer();
   
   if(NN_Load(g_h, path)) {
      Print("OK: Network loaded from: ", path);
      
      // Validace topologie
      int loaded_in = NN_InputSize(g_h);
      if(loaded_in != Lookback) {
         PrintFormat("WARNING: Loaded network InputSize=%d != Lookback=%d. Recreating...", loaded_in, Lookback);
         RecreateNet(); // Zahodit a vytvořit novou, jinak by spadlo forward
         return;
      }
      
      g_ready = true;
      PlaySound("Ok.wav");
      
      ForceRepaint(); // Okamžitě ukázat výsledek
      
   } else {
      Print("ERROR: Failed to load network. File not found? ", path);
      PlaySound("Timeout.wav");
   }
}

//+------------------------------------------------------------------+
//| GUI & EVENTS                                                     |
//+------------------------------------------------------------------+
void CreateGUI()
{
   int y = 50;
   int h = 25;
   int gap = 5;
   
   // Tlačítko MSE
   if(ObjectFind(0, g_btn_mse) < 0) {
      ObjectCreate(0, g_btn_mse, OBJ_BUTTON, 0, 0, 0);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_XSIZE, 120);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_YSIZE, h);
      ObjectSetString(0, g_btn_mse, OBJPROP_TEXT, "MSE Monitor");
      ObjectSetInteger(0, g_btn_mse, OBJPROP_BGCOLOR, clrDimGray);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_ZORDER, 10);
   }
   y += h + gap;

   // Tlačítko SAVE
   if(ObjectFind(0, g_btn_save) < 0) {
      ObjectCreate(0, g_btn_save, OBJ_BUTTON, 0, 0, 0);
      ObjectSetInteger(0, g_btn_save, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_btn_save, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, g_btn_save, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, g_btn_save, OBJPROP_XSIZE, 120);
      ObjectSetInteger(0, g_btn_save, OBJPROP_YSIZE, h);
      ObjectSetString(0, g_btn_save, OBJPROP_TEXT, "SAVE BIN");
      ObjectSetInteger(0, g_btn_save, OBJPROP_BGCOLOR, clrDodgerBlue);
      ObjectSetInteger(0, g_btn_save, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, g_btn_save, OBJPROP_ZORDER, 10);
   }
   y += h + gap;

   // Tlačítko LOAD
   if(ObjectFind(0, g_btn_load) < 0) {
      ObjectCreate(0, g_btn_load, OBJ_BUTTON, 0, 0, 0);
      ObjectSetInteger(0, g_btn_load, OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, g_btn_load, OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, g_btn_load, OBJPROP_YDISTANCE, y);
      ObjectSetInteger(0, g_btn_load, OBJPROP_XSIZE, 120);
      ObjectSetInteger(0, g_btn_load, OBJPROP_YSIZE, h);
      ObjectSetString(0, g_btn_load, OBJPROP_TEXT, "LOAD BIN");
      ObjectSetInteger(0, g_btn_load, OBJPROP_BGCOLOR, clrOrange);
      ObjectSetInteger(0, g_btn_load, OBJPROP_COLOR, clrWhite);
      ObjectSetInteger(0, g_btn_load, OBJPROP_ZORDER, 10);
   }
}

void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
{
   if(id == CHARTEVENT_OBJECT_CLICK) {
      // Toggle MSE
      if(sparam == g_btn_mse) {
         g_mse_visible = !g_mse_visible;
         NN_MSE_Show(g_mse_visible ? 1 : 0);
         ObjectSetInteger(0, g_btn_mse, OBJPROP_BGCOLOR, g_mse_visible ? clrLimeGreen : clrDimGray);
         ObjectSetInteger(0, g_btn_mse, OBJPROP_COLOR, g_mse_visible ? clrBlack : clrWhite);
         ChartRedraw();
      }
      
      // Save
      if(sparam == g_btn_save) {
         ObjectSetInteger(0, g_btn_save, OBJPROP_STATE, false); // Odkliknout
         SaveModelBin();
      }
      
      // Load
      if(sparam == g_btn_load) {
         ObjectSetInteger(0, g_btn_load, OBJPROP_STATE, false); // Odkliknout
         LoadModelBin();
      }
   }
}

//+------------------------------------------------------------------+
//| POMOCNÉ FUNKCE                                                   |
//+------------------------------------------------------------------+
void StartTimerSafe() {
   if(!g_timer_running) {
      EventSetMillisecondTimer(50); // 50ms interval pro trénink
      g_timer_running = true;
      if(DebugLog) Print(">>> Training Timer STARTED");
   }
}

void StopTimer() {
   if(g_timer_running) {
      EventKillTimer();
      g_timer_running = false;
      if(DebugLog) Print(">>> Training Timer STOPPED");
   }
}

void ForceRepaint()
{
   int total = Bars(_Symbol, _Period);
   if(total < TrainBars + Lookback) return;

   double o[], h[], l[], c[];
   datetime t[];
   
   if(CopyOpen(_Symbol, _Period, 0, total, o) < 0) return;
   if(CopyHigh(_Symbol, _Period, 0, total, h) < 0) return;
   if(CopyLow(_Symbol, _Period, 0, total, l) < 0) return;
   if(CopyClose(_Symbol, _Period, 0, total, c) < 0) return;
   if(CopyTime(_Symbol, _Period, 0, total, t) < 0) return;
   
   // Přepočítá historii i budoucnost
   CalculateHistory(o, h, l, c, total);
   DrawFuture(o, h, l, c, total, t);
   ChartRedraw();
}

double GetVal(int i, const double &o[], const double &h[], const double &l[], const double &c[]) {
   if(i < 0) return 0;
   switch(PriceMode) {
      case PRICE_OPEN: return o[i];
      case PRICE_HIGH: return h[i];
      case PRICE_LOW:  return l[i];
      case PRICE_MEDIAN: return (h[i]+l[i])*0.5;
      case PRICE_TYPICAL: return (h[i]+l[i]+c[i])/3.0;
      case PRICE_WEIGHTED: return (h[i]+l[i]+2*c[i])/4.0;
      default: return c[i];
   }
}

void RecreateNet() {
   if(g_h != 0) NN_Free(g_h);
   g_h = NN_Create();
   
   if(g_h > 0) {
      // Architektura sítě
      // 1. Vstupní vrstva -> Hidden
      NN_AddDense(g_h, Lookback, Hidden, HiddenAct); 
      // 2. Další skryté vrstvy
      for(int i=1; i<Depth; i++) NN_AddDense(g_h, Hidden, Hidden, HiddenAct);
      // 3. Výstupní vrstva (Linear pro regresi)
      // 3 = LINEAR
      NN_AddDense(g_h, Hidden, 1, 3);
   }
}

bool BuildDataset(const double &o[], const double &h[], const double &l[], const double &c[], int total)
{
   int start = total - TrainBars - ForecastHorizon - 1;
   if(start < Lookback + 1) return false;
   
   // 1. Výpočet průměru a směrodatné odchylky pro Z-Score normalizaci
   double sum = 0, sq_sum = 0;
   int count = 0;
   
   for(int i = start; i < total - ForecastHorizon; i++) {
      double val = GetVal(i, o,h,l,c);
      if(UseLogReturns) {
         double prev = GetVal(i-1, o,h,l,c);
         val = (prev > 0) ? MathLog(val / prev) : 0;
      }
      sum += val;
      sq_sum += val*val;
      count++;
   }
   
   g_mean = sum / count;
   double var = (sq_sum / count) - (g_mean * g_mean);
   g_std = (var > 0) ? MathSqrt(var) : 1.0;
   if(g_std < 1e-9) g_std = 1.0; 
   
   // 2. Naplnění vektorů pro DLL
   g_N = count - Lookback;
   if(g_N <= 0) return false;
   
   ArrayResize(g_X, g_N * Lookback);
   ArrayResize(g_T, g_N);
   
   int row = 0;
   for(int i = start + Lookback; i < total - ForecastHorizon; i++)
   {
      // Target
      double tgt_val = GetVal(i, o,h,l,c);
      if(UseLogReturns) {
         double p = GetVal(i-1, o,h,l,c);
         tgt_val = (p > 0) ? MathLog(tgt_val/p) : 0;
      }
      g_T[row] = (tgt_val - g_mean) / g_std;
      
      // Inputs
      for(int k = 0; k < Lookback; k++) {
         int idx = i - Lookback + k; 
         double x_val = GetVal(idx, o,h,l,c);
         if(UseLogReturns) {
            double p = GetVal(idx-1, o,h,l,c);
            x_val = (p > 0) ? MathLog(x_val/p) : 0;
         }
         g_X[row * Lookback + k] = (x_val - g_mean) / g_std;
      }
      row++;
   }
   return true;
}

void CalculateHistory(const double &o[], const double &h[], const double &l[], const double &c[], int total)
{
   int start = total - TrainBars; 
   if(start < Lookback + 10) start = Lookback + 10;
   
   double state[]; ArrayResize(state, Lookback);
   double out[1];
   
   for(int i = start; i < total; i++)
   {
      // Příprava vstupu
      for(int k = 0; k < Lookback; k++) {
         int idx = i - Lookback + 1 + k; // Posun o 1, abychom neviděli "budoucnost" (i)
         double val = GetVal(idx, o,h,l,c);
         if(UseLogReturns) {
            double p = GetVal(idx-1, o,h,l,c);
            val = (p > 0) ? MathLog(val/p) : 0;
         }
         state[k] = (val - g_mean) / g_std;
      }
      
      if(!NN_Forward(g_h, state, Lookback, out, 1)) break;
         
      // Denormalizace
      double net_out = out[0];
      double real_val = net_out * g_std + g_mean;
      double curr_price = GetVal(i, o,h,l,c);
      double pred_price = curr_price;
         
      if(UseLogReturns) {
          // Log return -> Cena
          pred_price = curr_price * MathExp(real_val);
      } 
      else {
          pred_price = real_val;
      }
      
      PredBuffer[i] = pred_price;
      
      // Target pro vizualizaci (posunuto o 1 bar dopředu)
      if(i + 1 < total)
         TgtBuffer[i] = GetVal(i + 1, o,h,l,c);
   }
}

void DrawFuture(const double &o[], const double &h[], const double &l[], const double &c[], int total, const datetime &tm[])
{
   if(!ShowFuture) return;
   ObjectsDeleteAll(0, "NNFUT_");
   
   double state[]; ArrayResize(state, Lookback);
   double out[1];
   
   // 1. Získat aktuální stav (posledních Lookback svíček)
   int now_idx = total - 1;
   for(int k = 0; k < Lookback; k++) {
      int idx = now_idx - Lookback + 1 + k;
      double val = GetVal(idx, o,h,l,c);
      if(UseLogReturns) {
         double p = GetVal(idx-1, o,h,l,c);
         val = (p > 0) ? MathLog(val/p) : 0;
      }
      state[k] = (val - g_mean) / g_std;
   }
   
   double curr_price = GetVal(now_idx, o,h,l,c);
   datetime curr_time = tm[now_idx];
   int period_sec = PeriodSeconds();
   
   // 2. Iterativní predikce
   for(int step = 1; step <= FuturePts; step++)
   {
      if(!NN_Forward(g_h, state, Lookback, out, 1)) break;
      
      double real_val = out[0] * g_std + g_mean;
      double next_price = curr_price;
      
      if(UseLogReturns) {
         next_price = curr_price * MathExp(real_val);
         // Posunout okno: vyhodit nejstarší, přidat predikci
         ArrayCopy(state, state, 0, 1, Lookback-1);
         state[Lookback-1] = out[0]; // Používáme predikovanou normalizovanou hodnotu jako vstup
      } else {
         next_price = real_val;
         ArrayCopy(state, state, 0, 1, Lookback-1);
         state[Lookback-1] = out[0];
      }
      
      datetime next_time = curr_time + period_sec;
      
      // Vykreslení segmentu
      string name = "NNFUT_" + IntegerToString(step);
      color seg_color = ColorFlat;
      if(next_price > curr_price) seg_color = ColorUp;
      else if(next_price < curr_price) seg_color = ColorDown;
      
      ObjectCreate(0, name, OBJ_TREND, 0, curr_time, curr_price, next_time, next_price);
      ObjectSetInteger(0, name, OBJPROP_COLOR, seg_color);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, WidthFuture);
      ObjectSetInteger(0, name, OBJPROP_RAY, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      
      curr_price = next_price;
      curr_time = next_time;
   }
}