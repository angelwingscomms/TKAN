bool AddMacdFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   double value = 0.0;
   if(
      !TkCalcMacdHist(symbol, timeframe, shift, CFG_MACD_FAST, CFG_MACD_SLOW, CFG_MACD_SIGNAL, value)
      || !TkPushValue(values, offset, value)
   )
      return false;
   return true;
}
