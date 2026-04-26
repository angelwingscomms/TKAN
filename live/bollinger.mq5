bool AddBollingerFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   for(int i = 0; i < ArraySize(CFG_BOLLINGER_PERIODS); i++)
   {
      int period = CFG_BOLLINGER_PERIODS[i];
      if(period <= 0)
         continue;
      double width = 0.0;
      double pctB = 0.0;
      if(
         !TkCalcBollinger(symbol, timeframe, shift, period, CFG_BOLLINGER_STD, width, pctB)
         || !TkPushValue(values, offset, width)
         || !TkPushValue(values, offset, pctB)
      )
         return false;
   }
   return true;
}
