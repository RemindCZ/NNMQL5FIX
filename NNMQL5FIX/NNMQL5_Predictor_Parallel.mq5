//+------------------------------------------------------------------+
//|                       NNMQL5_Predictor_BinSaveLoad_Realtime.mq5  |
//| V4.0: Realtime training + realtime redraw (mini-batch, no double)|
//| Vyžaduje: NNMQL5FIX.dll                                          |
//+------------------------------------------------------------------+
#property copyright "Remind"
#property link      "https://remind.cz"
#property version   "4.00"
//#property icon      "BPNDLL.ico"
#property description "MLP Predictor s realtime tréninkem a realtime vykreslením."

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

// --- Global VAR ---
bool g_shutting_down = false;


// --- DLL IMPORT ---
#import "NNMQL5FIX.dll"
int  NN_Create();
void NN_Free(int h);
bool NN_AddDense(int h, int inSz, int outSz, int act);
bool NN_Forward(int h, const double &in[], int in_len, double &out[], int out_len);
bool NN_TrainBatch(int h, const double &in[], int batch, int in_len, const double &tgt[], int tgt_len, double lr, double &mean_mse);
int  NN_InputSize(int h);

// Binary Save/Load functions
bool NN_Save(int h, string path);
bool NN_Load(int h, string path);

// MSE Window
void NN_MSE_Show(int show);
void NN_MSE_Push(double mse);
void NN_MSE_Clear();
void NN_MSE_SetMaxPoints(int n);
void NN_MSE_SetAutoScale(int enable, double y_min, double y_max);

// !!! CRITICAL FIX FOR CRASH !!!
void NN_GlobalCleanup();
#import

// --- VSTUPNÍ PARAMETRY ---
input group "Strategy & Data"
input bool     UseLogReturns   = true;      // TRUE = učí se změny
input int      Lookback        = 64;        // vstupní okno
input int      ForecastHorizon = 10;        // pro Target line (+1) / kompat.
input ENUM_APPLIED_PRICE PriceMode = PRICE_CLOSE;

input group "Network Architecture"
input int      Hidden          = 64;
input int      Depth           = 3;
input int      HiddenAct       = 2;         // 2=TANH
input double   LR              = 0.001;

input group "Training Control"
input int      TrainBars       = 2000;
input int      BatchSize       = 64;        // NYNÍ OPRAVDU POUŽITÉ
input int      BatchesPerTick  = 2;         // kolik minibatch kroků za jeden timer tick
input double   TargetMSE       = 0.000005;
input double   MaxMSEGuard     = 10.0;
input bool     UseTimer        = true;
input int      TimerMs         = 100;       // realtime, ale ne vražedné (50 ms je moc)
input bool     DebugLog        = true;

input group "File Operations"
input string   FileName        = "MyNet.bin";

input group "Visualization"
input bool     ShowFuture      = true;
input int      FuturePts       = 30;
input color    ColorUp         = clrLimeGreen;
input color    ColorDown       = clrTomato;
input color    ColorFlat       = clrSilver;
input int      WidthFuture     = 2;
input int      RealtimeBars    = 600;       // kolik posledních barů přepočítat pro realtime
input bool     RedrawEachTick  = true;      // ChartRedraw po každém timer kroku

// --- BUFFERS ---
double PredBuffer[];
double TgtBuffer[];

// --- GLOBALS ---
int    g_h = 0;
int    g_epochs = 0;
bool   g_ready = false;

double g_mean = 0.0;
double g_std  = 1.0;

double g_X[];  // dataset: g_N * Lookback
double g_T[];  // targets: g_N
int    g_N = 0;

bool   g_mse_visible = false;
bool   g_timer_running = false;

bool   g_busy = false;               // zámek proti reentranci
datetime g_last_bar_time = 0;        // detekce nové svíčky
int    g_last_rates_total = 0;

int    g_batch_cursor = 0;           // posuv po datasetu pro mini-batch

// minibatch buffery (alokované jednou)
double g_bX[];
double g_bT[];

// --- GUI NAMES ---
string g_btn_mse  = "Btn_ToggleMSE";
string g_btn_save = "Btn_SaveBin";
string g_btn_load = "Btn_LoadBin";

// --- PROTOTYPES ---
void CreateGUI();
void RecreateNet();
bool BuildDataset(const double &o[], const double &h[], const double &l[], const double &c[], int total);
void RecalcRealtimeWindow(const double &o[], const double &h[], const double &l[], const double &c[], const datetime &tm[], int total);
void DrawFuture(const double &o[], const double &h[], const double &l[], const double &c[], int total, const datetime &tm[]);
double GetVal(int i, const double &o[], const double &h[], const double &l[], const double &c[]);
void StartTimerSafe();
void StopTimer();
void SaveModelBin();
void LoadModelBin();

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool PrepareMiniBatch(int batch_sz);
bool TrainOneTick(double &mse_out);

void ForceRepaint();

//+------------------------------------------------------------------+
//| INIT                                                             |
//+------------------------------------------------------------------+
int OnInit()
  {
   MathSrand(GetTickCount());
   SetIndexBuffer(0, PredBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, TgtBuffer,  INDICATOR_DATA);
   PlotIndexSetInteger(0, PLOT_SHIFT, 1);
   PlotIndexSetInteger(1, PLOT_SHIFT, 1);
   PlotIndexSetDouble(0, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   PlotIndexSetDouble(1, PLOT_EMPTY_VALUE, EMPTY_VALUE);
   CreateGUI();
   RecreateNet();
   NN_MSE_SetMaxPoints(1000);
   NN_MSE_SetAutoScale(1, 0, 1);
   NN_MSE_Show(0);
// minibatch buffery
   ArrayResize(g_bX, MathMax(1, BatchSize) * Lookback);
   ArrayResize(g_bT, MathMax(1, BatchSize));
   if(UseTimer)
      StartTimerSafe();
   return INIT_SUCCEEDED;
  }

//+------------------------------------------------------------------+
//| DEINIT                                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // 0) Zabránit dalšímu tréninku
   g_shutting_down = true;

   // 1) Zastavit timer čistě přes tvoji funkci (nastaví i g_timer_running)
   StopTimer();

   // 2) Počkat, až doběhne případný právě běžící OnTimer/OnCalculate úsek
   //    (max ~500ms, ať se MT5 nezasekne)
   for(int i=0; i<50; i++)
   {
      if(!g_busy) break;
      Sleep(10);
   }

   // 3) Odstranit GUI a future objekty (už žádné volání do DLL tady)
   ObjectDelete(0, g_btn_mse);
   ObjectDelete(0, g_btn_save);
   ObjectDelete(0, g_btn_load);
   ObjectsDeleteAll(0, "NNFUT_");

   // 4) Zavřít MSE okno (pořád je to DLL call, ale ještě před globálním cleanupem)
   NN_MSE_Show(0);

   // 5) Kritické: globální cleanup (vypne UI thread + uvolní interní registry)
   //    Po tomhle už do DLL vůbec nevolej.
   NN_GlobalCleanup();

   // 6) Lokální handle jen vynulovat (nevolat NN_Free, aby nevznikl double-free)
   g_h = 0;

   Print("NNMQL5: Safe Cleanup Complete.");
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
   int need = TrainBars + Lookback + ForecastHorizon + 2;
   if(rates_total < need)
      return 0;
// neprovádíme trénink tady (realtime trénink je jen v OnTimer)
// tady jen detekujeme novou svíčku / změnu historie a případně rebuild datasetu
   if(g_busy)
      return rates_total;
   bool history_changed = (g_last_rates_total != 0 && rates_total != g_last_rates_total);
   bool new_bar = (rates_total > 1 && time[rates_total - 1] != g_last_bar_time);
   if(history_changed || new_bar || prev_calculated == 0)
     {
      g_busy = true;
      // dataset rebuild jen při změně
      if(BuildDataset(open, high, low, close, rates_total))
        {
         g_ready = true;
         // realtime přepočet posledního okna pro plynulý graf
         RecalcRealtimeWindow(open, high, low, close, time, rates_total);
         DrawFuture(open, high, low, close, rates_total, time);
        }
      g_last_rates_total = rates_total;
      g_last_bar_time = time[rates_total - 1];
      g_busy = false;
     }
   return rates_total;
  }

//+------------------------------------------------------------------+
//| TIMER - Realtime trénink + realtime redraw                        |
//+------------------------------------------------------------------+
void OnTimer()
  {
   if(g_shutting_down)
      return;
   if(g_busy)
      return;
   if(g_h == 0 || g_N <= 0)
      return;
   g_busy = true;
   double mse = 0.0;
   bool ok = TrainOneTick(mse);
   if(ok)
     {
      if(mse > MaxMSEGuard || !MathIsValidNumber(mse))
        {
         PrintFormat("CRITICAL: MSE Explosion (%.8f). Resetting network...", mse);
         RecreateNet();
         g_epochs = 0;
         g_ready = false;
         g_busy = false;
         return;
        }
      NN_MSE_Push(mse);
      g_epochs++;
      if(DebugLog && (g_epochs % 50 == 0))
        {
         PrintFormat("Train Epoch: %d | MSE: %.8f | N=%d | Batch=%d", g_epochs, mse, g_N, BatchSize);
        }
      // realtime repaint: přepočet jen posledního okna (rychlé)
      int total = Bars(_Symbol, _Period);
      if(total > 0)
        {
         double o[], h[], l[], c[];
         datetime t[];
         if(CopyOpen(_Symbol, _Period, 0, total, o) > 0 &&
            CopyHigh(_Symbol, _Period, 0, total, h) > 0 &&
            CopyLow(_Symbol, _Period, 0, total, l) > 0 &&
            CopyClose(_Symbol, _Period, 0, total, c) > 0 &&
            CopyTime(_Symbol, _Period, 0, total, t) > 0)
           {
            RecalcRealtimeWindow(o, h, l, c, t, total);
            DrawFuture(o, h, l, c, total, t);
            if(RedrawEachTick)
               ChartRedraw();
           }
        }
      if(mse < TargetMSE)
        {
         if(DebugLog)
            PrintFormat("SUCCESS: Target MSE %.8f reached. Training paused.", TargetMSE);
         StopTimer();
         if(RedrawEachTick)
            ChartRedraw();
         g_busy = false;
         return;
        }
     }
   g_busy = false;
  }

//+------------------------------------------------------------------+
//| ULOŽENÍ BINÁRNÍ SÍTĚ                                             |
//+------------------------------------------------------------------+
void SaveModelBin()
  {
   if(g_h <= 0)
     {
      Print("Warning: Network not valid.");
      return;
     }
   string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + FileName;
   if(NN_Save(g_h, path))
     {
      Print("OK: Network saved to: ", path);
      PlaySound("Ok.wav");
     }
   else
     {
      Print("ERROR: Failed to save network to: ", path);
      PlaySound("Timeout.wav");
     }
  }

//+------------------------------------------------------------------+
//| NAČTENÍ BINÁRNÍ SÍTĚ                                             |
//+------------------------------------------------------------------+
void LoadModelBin()
  {
   if(g_h <= 0)
      RecreateNet();
   string path = TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + FileName;
   StopTimer();
   if(NN_Load(g_h, path))
     {
      Print("OK: Network loaded from: ", path);
      int loaded_in = NN_InputSize(g_h);
      if(loaded_in != Lookback)
        {
         PrintFormat("WARNING: Loaded network InputSize=%d, Lookback=%d (může selhat forward).", loaded_in, Lookback);
        }
      g_ready = true;
      PlaySound("Ok.wav");
      // po loadu: přepočítat graf hned (realtime)
      ForceRepaint();
     }
   else
     {
      Print("ERROR: Failed to load network. Does the file exist? ", path);
      PlaySound("Timeout.wav");
     }
  }

//+------------------------------------------------------------------+
//| GUI & EVENTS                                                     |
//+------------------------------------------------------------------+
void CreateGUI()
  {
   int y = 50, h = 25, gap = 5;
   if(ObjectFind(0, g_btn_mse) < 0)
     {
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
   if(ObjectFind(0, g_btn_save) < 0)
     {
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
   if(ObjectFind(0, g_btn_load) < 0)
     {
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

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void OnChartEvent(const int id, const long &lparam, const double &dparam, const string &sparam)
  {
   if(id != CHARTEVENT_OBJECT_CLICK)
      return;
   if(sparam == g_btn_mse)
     {
      g_mse_visible = !g_mse_visible;
      NN_MSE_Show(g_mse_visible ? 1 : 0);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_BGCOLOR, g_mse_visible ? clrLimeGreen : clrDimGray);
      ObjectSetInteger(0, g_btn_mse, OBJPROP_COLOR, g_mse_visible ? clrBlack : clrWhite);
      ChartRedraw();
      return;
     }
   if(sparam == g_btn_save)
     {
      ObjectSetInteger(0, g_btn_save, OBJPROP_STATE, false);
      SaveModelBin();
      return;
     }
   if(sparam == g_btn_load)
     {
      ObjectSetInteger(0, g_btn_load, OBJPROP_STATE, false);
      LoadModelBin();
      return;
     }
  }

//+------------------------------------------------------------------+
//| Timer helpers                                                    |
//+------------------------------------------------------------------+
void StartTimerSafe()
  {
   if(!g_timer_running)
     {
      EventSetMillisecondTimer(MathMax(20, TimerMs));
      g_timer_running = true;
      if(DebugLog)
         Print(">>> Training Timer STARTED");
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void StopTimer()
  {
   if(g_timer_running)
     {
      EventKillTimer();
      g_timer_running = false;
      if(DebugLog)
         Print(">>> Training Timer STOPPED");
     }
  }

//+------------------------------------------------------------------+
//| Repaint (full)                                                   |
//+------------------------------------------------------------------+
void ForceRepaint()
  {
   int total = Bars(_Symbol, _Period);
   if(total < TrainBars + Lookback + 5)
      return;
   double o[], h[], l[], c[];
   datetime t[];
   if(CopyOpen(_Symbol, _Period, 0, total, o) <= 0)
      return;
   if(CopyHigh(_Symbol, _Period, 0, total, h) <= 0)
      return;
   if(CopyLow(_Symbol, _Period, 0, total, l) <= 0)
      return;
   if(CopyClose(_Symbol, _Period, 0, total, c) <= 0)
      return;
   if(CopyTime(_Symbol, _Period, 0, total, t) <= 0)
      return;
// dataset rebuild (pro jistotu)
   BuildDataset(o, h, l, c, total);
   ArrayInitialize(PredBuffer, EMPTY_VALUE);
   ArrayInitialize(TgtBuffer,  EMPTY_VALUE);
   RecalcRealtimeWindow(o, h, l, c, t, total);
   DrawFuture(o, h, l, c, total, t);
   ChartRedraw();
  }

//+------------------------------------------------------------------+
//| Dataset + Normalizace                                            |
//+------------------------------------------------------------------+
double GetVal(int i, const double &o[], const double &h[], const double &l[], const double &c[])
  {
   if(i < 0)
      return 0.0;
   switch(PriceMode)
     {
      case PRICE_OPEN:
         return o[i];
      case PRICE_HIGH:
         return h[i];
      case PRICE_LOW:
         return l[i];
      case PRICE_MEDIAN:
         return (h[i] + l[i]) * 0.5;
      case PRICE_TYPICAL:
         return (h[i] + l[i] + c[i]) / 3.0;
      case PRICE_WEIGHTED:
         return (h[i] + l[i] + 2 * c[i]) / 4.0;
      default:
         return c[i];
     }
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
void RecreateNet()
  {
   if(g_h != 0)
      NN_Free(g_h);
   g_h = NN_Create();
   NN_AddDense(g_h, Lookback, Hidden, HiddenAct);
   for(int i = 1; i < Depth; i++)
      NN_AddDense(g_h, Hidden, Hidden, HiddenAct);
   NN_AddDense(g_h, Hidden, 1, 3); // LINEAR
   g_ready = false;
   g_epochs = 0;
   g_batch_cursor = 0;
   NN_MSE_Clear();
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool BuildDataset(const double &o[], const double &h[], const double &l[], const double &c[], int total)
  {
   int start = total - TrainBars - ForecastHorizon - 1;
   if(start < Lookback + 2)
      return false;
   double sum = 0.0, sq_sum = 0.0;
   int count = 0;
   for(int i = start; i < total - ForecastHorizon; i++)
     {
      double val = GetVal(i, o, h, l, c);
      if(UseLogReturns)
        {
         double prev = GetVal(i - 1, o, h, l, c);
         val = (prev > 0.0 && val > 0.0) ? MathLog(val / prev) : 0.0;
        }
      sum += val;
      sq_sum += val * val;
      count++;
     }
   if(count <= Lookback + 1)
      return false;
   g_mean = sum / count;
   double var = (sq_sum / count) - (g_mean * g_mean);
   g_std = (var > 1e-18) ? MathSqrt(var) : 1.0;
   if(g_std < 1e-9)
      g_std = 1.0;
   g_N = count - Lookback;
   if(g_N <= 0)
      return false;
   ArrayResize(g_X, g_N * Lookback);
   ArrayResize(g_T, g_N);
   int row = 0;
   for(int i = start + Lookback; i < total - ForecastHorizon; i++)
     {
      double tgt_val = GetVal(i, o, h, l, c);
      if(UseLogReturns)
        {
         double p = GetVal(i - 1, o, h, l, c);
         tgt_val = (p > 0.0 && tgt_val > 0.0) ? MathLog(tgt_val / p) : 0.0;
        }
      g_T[row] = (tgt_val - g_mean) / g_std;
      for(int k = 0; k < Lookback; k++)
        {
         int idx = i - Lookback + k;
         double x_val = GetVal(idx, o, h, l, c);
         if(UseLogReturns)
           {
            double p = GetVal(idx - 1, o, h, l, c);
            x_val = (p > 0.0 && x_val > 0.0) ? MathLog(x_val / p) : 0.0;
           }
         g_X[row * Lookback + k] = (x_val - g_mean) / g_std;
        }
      row++;
     }
// připrav minibatch buffery pro aktuální BatchSize
   int bs = MathMax(1, BatchSize);
   ArrayResize(g_bX, bs * Lookback);
   ArrayResize(g_bT, bs);
// aby cursor nevylítl mimo
   if(g_batch_cursor >= g_N)
      g_batch_cursor = 0;
   return true;
  }

//+------------------------------------------------------------------+
//| Mini-batch (sekvenční cursor, realtime-friendly)                  |
//+------------------------------------------------------------------+
bool PrepareMiniBatch(int batch_sz)
  {
   if(g_N <= 0)
      return false;
   if(batch_sz <= 0)
      return false;
   int bs = batch_sz;
   if(bs > g_N)
      bs = g_N;
// pokud dojedeme na konec, zatočíme se (realtime bez shuffle overhead)
   for(int r = 0; r < bs; r++)
     {
      int row = g_batch_cursor++;
      if(g_batch_cursor >= g_N)
         g_batch_cursor = 0;
      // X
      int src = row * Lookback;
      int dst = r * Lookback;
      for(int k = 0; k < Lookback; k++)
         g_bX[dst + k] = g_X[src + k];
      // T
      g_bT[r] = g_T[row];
     }
// zmenši pole pro DLL call (MQL5 předává celé pole, ale batch určuje délku)
   return true;
  }

//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
bool TrainOneTick(double &mse_out)
  {
   mse_out = 0.0;
   if(g_h == 0 || g_N <= 0)
      return false;
   int bs = MathMax(1, BatchSize);
   if(bs > g_N)
      bs = g_N;
   double mse_acc = 0.0;
   int steps = MathMax(1, BatchesPerTick);
   for(int s = 0; s < steps; s++)
     {
      if(!PrepareMiniBatch(bs))
         return false;
      double mse = 0.0;
      bool ok = NN_TrainBatch(g_h, g_bX, bs, Lookback, g_bT, 1, LR, mse);
      if(!ok)
         return false;
      mse_acc += mse;
     }
   mse_out = mse_acc / steps;
   g_ready = true;
   return true;
  }

//+------------------------------------------------------------------+
//| Realtime výpočet křivek: jen poslední okno                         |
//+------------------------------------------------------------------+
void RecalcRealtimeWindow(const double &o[], const double &h[], const double &l[], const double &c[],
                          const datetime &tm[], int total)
  {
   if(!g_ready || g_h == 0)
      return;
   int start = total - MathMax(RealtimeBars, Lookback + 5);
   if(start < Lookback + 2)
      start = Lookback + 2;
// nevynulovat celé buffery (to je drahé), jen přepsat okno
   double state[];
   ArrayResize(state, Lookback);
   double out[1];
   for(int i = start; i < total; i++)
     {
      for(int k = 0; k < Lookback; k++)
        {
         int idx = i - Lookback + 1 + k;
         double val = GetVal(idx, o, h, l, c);
         if(UseLogReturns)
           {
            double p = GetVal(idx - 1, o, h, l, c);
            val = (p > 0.0 && val > 0.0) ? MathLog(val / p) : 0.0;
           }
         state[k] = (val - g_mean) / g_std;
        }
      if(!NN_Forward(g_h, state, Lookback, out, 1))
        {
         PredBuffer[i] = EMPTY_VALUE;
         continue;
        }
      double real_val = out[0] * g_std + g_mean;
      double curr_price = GetVal(i, o, h, l, c);
      double pred_price = curr_price;
      if(UseLogReturns)
         pred_price = curr_price * MathExp(real_val);
      else
         pred_price = real_val;
      PredBuffer[i] = pred_price;
      // Target +1 (realtime)
      if(i + 1 < total)
         TgtBuffer[i] = GetVal(i + 1, o, h, l, c);
      else
         TgtBuffer[i] = EMPTY_VALUE;
     }
  }

//+------------------------------------------------------------------+
//| Future trendlines (realtime)                                     |
//+------------------------------------------------------------------+
void DrawFuture(const double &o[], const double &h[], const double &l[], const double &c[], int total, const datetime &tm[])
  {
   if(!ShowFuture || !g_ready || g_h == 0)
      return;
   ObjectsDeleteAll(0, "NNFUT_");
   double state[];
   ArrayResize(state, Lookback);
   double out[1];
   int now_idx = total - 1;
   for(int k = 0; k < Lookback; k++)
     {
      int idx = now_idx - Lookback + 1 + k;
      double val = GetVal(idx, o, h, l, c);
      if(UseLogReturns)
        {
         double p = GetVal(idx - 1, o, h, l, c);
         val = (p > 0.0 && val > 0.0) ? MathLog(val / p) : 0.0;
        }
      state[k] = (val - g_mean) / g_std;
     }
   double curr_price = GetVal(now_idx, o, h, l, c);
   datetime curr_time = tm[now_idx];
   int period_sec = PeriodSeconds();
   for(int step = 1; step <= FuturePts; step++)
     {
      if(!NN_Forward(g_h, state, Lookback, out, 1))
         break;
      double real_val = out[0] * g_std + g_mean;
      double next_price = curr_price;
      if(UseLogReturns)
        {
         next_price = curr_price * MathExp(real_val);
         // posun okna: přidáváme predikovaný normalizovaný výstup
         ArrayCopy(state, state, 0, 1, Lookback - 1);
         state[Lookback - 1] = out[0];
        }
      else
        {
         next_price = real_val;
         ArrayCopy(state, state, 0, 1, Lookback - 1);
         state[Lookback - 1] = out[0];
        }
      datetime next_time = curr_time + period_sec;
      string name = "NNFUT_" + IntegerToString(step);
      color seg_color = ColorFlat;
      if(next_price > curr_price)
         seg_color = ColorUp;
      else
         if(next_price < curr_price)
            seg_color = ColorDown;
      ObjectCreate(0, name, OBJ_TREND, 0, curr_time, curr_price, next_time, next_price);
      ObjectSetInteger(0, name, OBJPROP_COLOR, seg_color);
      ObjectSetInteger(0, name, OBJPROP_WIDTH, WidthFuture);
      ObjectSetInteger(0, name, OBJPROP_RAY, false);
      ObjectSetInteger(0, name, OBJPROP_SELECTABLE, false);
      curr_price = next_price;
      curr_time  = next_time;
     }
  }
//+------------------------------------------------------------------+

