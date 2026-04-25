#property copyright "Simple"
#property version "1.00"
#property script_show_inputs

input string InpStartDate = "2021.05.01";
input string InpEndDate = "2021.07.01";

void OnStart() {
   datetime startTime = StringToTime(InpStartDate);
   datetime endTime = StringToTime(InpEndDate);
   
   Print("Fetching BTCUSD data from ", InpStartDate, " to ", InpEndDate);
   
   string filename = "data.csv";
   int fileHandle = FileOpen(filename, FILE_CSV|FILE_WRITE, ",");
   
   if(fileHandle == INVALID_HANDLE) {
      Print("Failed to open file: ", GetLastError());
      return;
   }
   
   FileWrite(fileHandle, "datetime,open,high,low,close");
   
   MqlRates rates[];
   ArraySetAsSeries(rates, true);
   
   int copied = CopyRates("BTCUSD", PERIOD_H1, startTime, endTime, rates);
   
   if(copied <= 0) {
      Print("Failed to copy rates: ", GetLastError());
      FileClose(fileHandle);
      return;
   }
   
   Print("Copied ", copied, " bars");
   
   for(int i = 0; i < copied; i++) {
      string dtStr = TimeToString(rates[i].time, TIME_DATE|TIME_MINUTES);
      StringReplace(dtStr, ".", "-");
      StringReplace(dtStr, ":", "-");
      
      string line = dtStr + "," +
                  DoubleToString(rates[i].open, 2) + "," +
                  DoubleToString(rates[i].high, 2) + "," +
                  DoubleToString(rates[i].low, 2) + "," +
                  DoubleToString(rates[i].close, 2);
      
      FileWrite(fileHandle, line);
   }
   
   FileClose(fileHandle);
   
   Print("Successfully written ", copied, " rows to ", filename);
}