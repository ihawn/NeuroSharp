using NeuroSharp.Data;
using NeuroSharp.Datatypes;

namespace Trainer
{
    public class Trainer
    {
        static void Main(string[] args)
        {
            LetterIdentificationTraining(5);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            ImageDataAggregate data = ImagePreprocessor.GetImageData(
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters\english.csv",
                @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\english handwritten characters",
                expectedHeight: 900, expectedWidth: 1200
            );
        }
    }
}