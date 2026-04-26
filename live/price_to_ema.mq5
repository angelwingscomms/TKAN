bool AddPriceToEmaFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   double close = iClose(symbol, timeframe, shift);
   if(close <= TkEps())
      return false;
   for(int i = 0; i < ArraySize(CFG_PRICE_TO_EMA_PERIODS); i++)
   {
      int period = CFG_PRICE_TO_EMA_PERIODS[i];
      if(period <= 0)
         continue;
      double average = 0.0;
      if(!TkCalcEma(symbol, timeframe, shift, period, average) || !TkPushValue(values, offset, TkRatioMinusOne(close, average)))
         return false;
   }
   return true;
}
