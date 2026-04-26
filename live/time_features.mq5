bool AddTimeFeatures(const string anchorSymbol, const int barShift, double &values[], int &offset)
{
   datetime openTime = iTime(anchorSymbol, PERIOD_CURRENT, barShift);
   if(openTime == 0)
      return false;
   datetime closeTime = openTime + PeriodSeconds(PERIOD_CURRENT);
   MqlDateTime ts;
   TimeToStruct(closeTime, ts);

   if(CFG_TIME_HOUR_ENABLED)
   {
      double angle = 2.0 * 3.14159265358979323846 * ts.hour / 24.0;
      if(!TkPushValue(values, offset, MathSin(angle)) || !TkPushValue(values, offset, MathCos(angle)))
         return false;
   }

   if(CFG_TIME_MINUTE_ENABLED)
   {
      double angle = 2.0 * 3.14159265358979323846 * ts.min / 60.0;
      if(!TkPushValue(values, offset, MathSin(angle)) || !TkPushValue(values, offset, MathCos(angle)))
         return false;
   }

   if(CFG_TIME_DAY_OF_WEEK_ENABLED)
   {
      double dayOfWeek = ((ts.day_of_week + 6) % 7) / 6.0;
      if(!TkPushValue(values, offset, dayOfWeek))
         return false;
   }

   return true;
}
