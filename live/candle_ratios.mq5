bool AddCandleRatioFeatures(const string symbol, const ENUM_TIMEFRAMES timeframe, const int shift, double &values[], int &offset)
{
   double open = iOpen(symbol, timeframe, shift);
   double high = iHigh(symbol, timeframe, shift);
   double low = iLow(symbol, timeframe, shift);
   double close = iClose(symbol, timeframe, shift);
   if(open <= TkEps() || high <= TkEps() || low <= TkEps() || close <= TkEps())
      return false;
   return TkPushValue(values, offset, TkRatioMinusOne(close, open))
      && TkPushValue(values, offset, TkRatioMinusOne(high, low));
}
