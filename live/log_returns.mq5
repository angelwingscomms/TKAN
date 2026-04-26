bool AddLogReturnFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   for(int i = 0; i < ArraySize(CFG_LOG_RETURN_PERIODS); i++)
   {
      int period = CFG_LOG_RETURN_PERIODS[i];
      if(period <= 0)
         continue;
      double value = 0.0;
      if(!TkCalcLogReturn(symbol, timeframe, shift, period, value) || !TkPushValue(values, offset, value))
         return false;
   }
   return true;
}
