bool AddHigherTimeframeFeatures(const string anchorSymbol, const string symbol, const int barShift, double &values[], int &offset)
{
   for(int i = 0; i < ArraySize(CFG_HIGHER_TIMEFRAME_MINUTES); i++)
   {
      int minutes = CFG_HIGHER_TIMEFRAME_MINUTES[i];
      if(minutes <= 0)
         continue;
      ENUM_TIMEFRAMES timeframe = TkMinutesToTimeframe(minutes);
      int shift = TkCompletedHigherShift(anchorSymbol, symbol, minutes, barShift);
      if(shift < 0)
         return false;

      for(int j = 0; j < ArraySize(CFG_HIGHER_TIMEFRAME_LOG_RETURN_PERIODS); j++)
      {
         int period = CFG_HIGHER_TIMEFRAME_LOG_RETURN_PERIODS[j];
         if(period <= 0)
            continue;
         double logReturn = 0.0;
         if(!TkCalcLogReturn(symbol, timeframe, shift, period, logReturn) || !TkPushValue(values, offset, logReturn))
            return false;
      }

      for(int j = 0; j < ArraySize(CFG_HIGHER_TIMEFRAME_RSI_PERIODS); j++)
      {
         int period = CFG_HIGHER_TIMEFRAME_RSI_PERIODS[j];
         if(period <= 0)
            continue;
         double rsi = 0.0;
         if(!TkCalcRsi(symbol, timeframe, shift, period, rsi) || !TkPushValue(values, offset, rsi))
            return false;
      }

      double macd = 0.0;
      if(
         !TkCalcMacdHist(
            symbol,
            timeframe,
            shift,
            CFG_HIGHER_TIMEFRAME_MACD_FAST,
            CFG_HIGHER_TIMEFRAME_MACD_SLOW,
            CFG_HIGHER_TIMEFRAME_MACD_SIGNAL,
            macd
         )
         || !TkPushValue(values, offset, macd)
      )
         return false;
   }
   return true;
}
