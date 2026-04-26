double TkEps()
{
   return 1e-12;
}

bool TkPushValue(double &values[], int &offset, double value)
{
   if(!MathIsValidNumber(value) || offset >= ArraySize(values))
      return false;
   values[offset++] = value;
   return true;
}

double TkRatioMinusOne(double numerator, double denominator)
{
   if(denominator <= TkEps())
      return 0.0;
   return numerator / denominator - 1.0;
}

ENUM_TIMEFRAMES TkMinutesToTimeframe(int minutes)
{
   switch(minutes)
   {
      case 1: return PERIOD_M1;
      case 5: return PERIOD_M5;
      case 15: return PERIOD_M15;
      case 30: return PERIOD_M30;
      case 60: return PERIOD_H1;
      case 240: return PERIOD_H4;
      case 1440: return PERIOD_D1;
   }
   return PERIOD_CURRENT;
}

int TkCompletedHigherShift(const string anchorSymbol, const string symbol, const int minutes, const int barShift)
{
   ENUM_TIMEFRAMES timeframe = TkMinutesToTimeframe(minutes);
   datetime barTime = iTime(anchorSymbol, PERIOD_CURRENT, barShift);
   if(barTime == 0)
      return -1;
   datetime closeTime = barTime + PeriodSeconds(PERIOD_CURRENT);
   int shift = iBarShift(symbol, timeframe, closeTime, false);
   if(shift < 0)
      return -1;
   return shift + 1;
}

bool TkCalcLogReturn(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, double &value)
{
   double currentClose = iClose(symbol, timeframe, shift);
   double previousClose = iClose(symbol, timeframe, shift + period);
   if(currentClose <= TkEps() || previousClose <= TkEps())
      return false;
   value = MathLog(currentClose / previousClose);
   return MathIsValidNumber(value);
}

bool TkCalcRollingStdLogReturn(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int window, double &value)
{
   double sum = 0.0;
   double sumSq = 0.0;
   for(int i = 0; i < window; i++)
   {
      double ret = 0.0;
      if(!TkCalcLogReturn(symbol, timeframe, shift + i, 1, ret))
         return false;
      sum += ret;
      sumSq += ret * ret;
   }
   double mean = sum / window;
   double variance = sumSq / window - mean * mean;
   if(variance < 0.0)
      variance = 0.0;
   value = MathSqrt(variance);
   return true;
}

bool TkCalcGarmanKlass(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int window, double &value)
{
   double total = 0.0;
   double coeff = 2.0 * MathLog(2.0) - 1.0;
   for(int i = 0; i < window; i++)
   {
      int bar = shift + i;
      double open = iOpen(symbol, timeframe, bar);
      double high = iHigh(symbol, timeframe, bar);
      double low = iLow(symbol, timeframe, bar);
      double close = iClose(symbol, timeframe, bar);
      if(open <= TkEps() || high <= TkEps() || low <= TkEps() || close <= TkEps())
         return false;
      double logHL = MathLog(high / low);
      double logCO = MathLog(close / open);
      double variance = 0.5 * logHL * logHL - coeff * logCO * logCO;
      if(variance < 0.0)
         variance = 0.0;
      total += variance;
   }
   value = MathSqrt(total / window);
   return true;
}

bool TkCalcBollinger(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, const double numStd, double &width, double &pctB)
{
   double sum = 0.0;
   double sumSq = 0.0;
   for(int i = 0; i < period; i++)
   {
      double close = iClose(symbol, timeframe, shift + i);
      if(close <= TkEps())
         return false;
      sum += close;
      sumSq += close * close;
   }
   double mean = sum / period;
   double variance = sumSq / period - mean * mean;
   if(variance < 0.0)
      variance = 0.0;
   double std = MathSqrt(variance);
   double upper = mean + numStd * std;
   double lower = mean - numStd * std;
   double latestClose = iClose(symbol, timeframe, shift);
   if(mean <= TkEps() || upper <= lower)
      return false;
   width = (upper - lower) / mean;
   pctB = (latestClose - lower) / (upper - lower);
   return MathIsValidNumber(width) && MathIsValidNumber(pctB);
}

bool TkCalcSma(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, double &value)
{
   double sum = 0.0;
   for(int i = 0; i < period; i++)
   {
      double close = iClose(symbol, timeframe, shift + i);
      if(close <= TkEps())
         return false;
      sum += close;
   }
   value = sum / period;
   return true;
}

bool TkCalcEma(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, double &value)
{
   int oldest = shift + period + 200;
   double ema = iClose(symbol, timeframe, oldest);
   if(ema <= TkEps())
      return false;
   double alpha = 2.0 / (period + 1.0);
   for(int bar = oldest - 1; bar >= shift; bar--)
   {
      double close = iClose(symbol, timeframe, bar);
      if(close <= TkEps())
         return false;
      ema = alpha * close + (1.0 - alpha) * ema;
   }
   value = ema;
   return true;
}

bool TkCalcRsi(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, double &value)
{
   int oldest = shift + period + 200;
   double previousClose = iClose(symbol, timeframe, oldest);
   if(previousClose <= TkEps())
      return false;

   double avgGain = 0.0;
   double avgLoss = 0.0;
   int count = 0;

   for(int bar = oldest - 1; bar >= shift; bar--)
   {
      double close = iClose(symbol, timeframe, bar);
      if(close <= TkEps())
         return false;
      double delta = close - previousClose;
      double gain = delta > 0.0 ? delta : 0.0;
      double loss = delta < 0.0 ? -delta : 0.0;
      if(count < period)
      {
         avgGain += gain;
         avgLoss += loss;
         if(count == period - 1)
         {
            avgGain /= period;
            avgLoss /= period;
         }
      }
      else
      {
         avgGain = ((period - 1.0) * avgGain + gain) / period;
         avgLoss = ((period - 1.0) * avgLoss + loss) / period;
      }
      previousClose = close;
      count++;
   }

   if(count < period)
      return false;
   double rs = avgGain / (avgLoss + TkEps());
   value = 100.0 - 100.0 / (1.0 + rs);
   return MathIsValidNumber(value);
}

bool TkCalcAdx(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int period, double &value)
{
   int oldest = shift + period * 3 + 200;
   double prevHigh = iHigh(symbol, timeframe, oldest);
   double prevLow = iLow(symbol, timeframe, oldest);
   double prevClose = iClose(symbol, timeframe, oldest);
   if(prevHigh <= TkEps() || prevLow <= TkEps() || prevClose <= TkEps())
      return false;

   double atr = 0.0;
   double plusDmSmooth = 0.0;
   double minusDmSmooth = 0.0;
   double adx = 0.0;
   double dxSeed = 0.0;
   int trCount = 0;
   int dxCount = 0;

   for(int bar = oldest - 1; bar >= shift; bar--)
   {
      double high = iHigh(symbol, timeframe, bar);
      double low = iLow(symbol, timeframe, bar);
      double close = iClose(symbol, timeframe, bar);
      if(high <= TkEps() || low <= TkEps() || close <= TkEps())
         return false;

      double upMove = high - prevHigh;
      double downMove = prevLow - low;
      double plusDm = (upMove > downMove && upMove > 0.0) ? upMove : 0.0;
      double minusDm = (downMove > upMove && downMove > 0.0) ? downMove : 0.0;
      double tr = MathMax(high - low, MathMax(MathAbs(high - prevClose), MathAbs(low - prevClose)));

      if(trCount < period)
      {
         atr += tr;
         plusDmSmooth += plusDm;
         minusDmSmooth += minusDm;
         if(trCount == period - 1)
         {
            atr /= period;
            plusDmSmooth /= period;
            minusDmSmooth /= period;
         }
      }
      else
      {
         atr = ((period - 1.0) * atr + tr) / period;
         plusDmSmooth = ((period - 1.0) * plusDmSmooth + plusDm) / period;
         minusDmSmooth = ((period - 1.0) * minusDmSmooth + minusDm) / period;
      }

      if(trCount >= period - 1)
      {
         double plusDi = 100.0 * plusDmSmooth / (atr + TkEps());
         double minusDi = 100.0 * minusDmSmooth / (atr + TkEps());
         double dx = 100.0 * MathAbs(plusDi - minusDi) / (plusDi + minusDi + TkEps());
         if(dxCount < period)
         {
            dxSeed += dx;
            if(dxCount == period - 1)
               adx = dxSeed / period;
         }
         else
         {
            adx = ((period - 1.0) * adx + dx) / period;
         }
         dxCount++;
      }

      prevHigh = high;
      prevLow = low;
      prevClose = close;
      trCount++;
   }

   if(dxCount < period)
      return false;
   value = adx;
   return MathIsValidNumber(value);
}

bool TkCalcMacdHist(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, const int fast, const int slow, const int signal, double &value)
{
   int oldest = shift + slow + signal + 200;
   double emaFast = iClose(symbol, timeframe, oldest);
   double emaSlow = emaFast;
   if(emaFast <= TkEps())
      return false;
   double signalValue = 0.0;
   bool signalReady = false;
   double alphaFast = 2.0 / (fast + 1.0);
   double alphaSlow = 2.0 / (slow + 1.0);
   double alphaSignal = 2.0 / (signal + 1.0);
   double latestClose = iClose(symbol, timeframe, shift);
   if(latestClose <= TkEps())
      return false;

   for(int bar = oldest - 1; bar >= shift; bar--)
   {
      double close = iClose(symbol, timeframe, bar);
      if(close <= TkEps())
         return false;
      emaFast = alphaFast * close + (1.0 - alphaFast) * emaFast;
      emaSlow = alphaSlow * close + (1.0 - alphaSlow) * emaSlow;
      double macd = emaFast - emaSlow;
      if(!signalReady)
      {
         signalValue = macd;
         signalReady = true;
      }
      else
      {
         signalValue = alphaSignal * macd + (1.0 - alphaSignal) * signalValue;
      }
   }

   value = (emaFast - emaSlow - signalValue) / latestClose;
   return MathIsValidNumber(value);
}
