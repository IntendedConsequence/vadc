#include <stdio.h>

// extern "C" void mainCRTStartup()
int main()
{
   fputs("asetpts=N/SR/TB, aselect='", stdout);

   float from, to;
   int lines_processed = 0;
   while (scanf_s("%f,%f", &from, &to) != EOF)
   {
      if (lines_processed > 0)
      {
         fputc('+', stdout);
      }
      fprintf(stdout, "between(t,%f,%f)", from, to);

      ++lines_processed;
   }
   fputs("', asetpts=N/SR/TB", stdout);

   return 0;
}

/*
      sys.stdout.write("aselect='")
      for i, speech_dict in enumerate(get_speech_timestamps_stdin(None, model, return_seconds=True)):
         if i:
            sys.stdout.write("+")
         sys.stdout.write("between(t,{},{})".format(speech_dict['start'], speech_dict['end']))
         #print(speech_dict['start'], speech_dict['end'])
      sys.stdout.write("', asetpts=N/SR/TB")
*/
