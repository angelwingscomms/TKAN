bool AddEmaCrossFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   for(int i = 0; i < ArraySize(CFG_EMA_CROSS_FAST); i++)
   {
      int fast = CFG_EMA_CROSS_FAST[i];
      int slow = CFG_EMA_CROSS_SLOW[i];
      if(fast <= 0 || slow <= 0)
         continue;
      double fastValue = 0.0;
      double slowValue = 0.0;
      if(
         !TkCalcEma(symbol, timeframe, shift, fast, fastValue)
         || !TkCalcEma(symbol, timeframe, shift, slow, slowValue)
         || !TkPushValue(values, offset, TkRatioMinusOne(fastValue, slowValue))
      )
         return false;
   }
   return true;
}
