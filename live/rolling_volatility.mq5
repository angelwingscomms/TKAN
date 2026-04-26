bool AddRollingVolatilityFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   for(int i = 0; i < ArraySize(CFG_ROLLING_VOLATILITY_WINDOWS); i++)
   {
      int window = CFG_ROLLING_VOLATILITY_WINDOWS[i];
      if(window <= 0)
         continue;
      double value = 0.0;
      if(!TkCalcRollingStdLogReturn(symbol, timeframe, shift, window, value) || !TkPushValue(values, offset, value))
         return false;
   }
   return true;
}
