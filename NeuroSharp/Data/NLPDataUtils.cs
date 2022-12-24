using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;


namespace NeuroSharp.Data
{
    public static class NLPDataUtils
    {
        public static Vector<double> GetEncodedWordStream(string sentence, List<string> wordList, int maxLength)
        {
            Vector<double> x = Vector<double>.Build.Dense(wordList.Count * maxLength);

            string allowedChars = "abcdefghijklmnopqrstuvwxyz ";
            List<string> reviewWords = new string(sentence.ToLower()
                    .Replace(".", " ")
                    .Replace(",", " ")
                    .Where(c => allowedChars.Contains(c)).ToArray())
                .Split(' ')
                .Where(s => wordList.Contains(s)).ToList();

            for (int j = 0; j < Math.Min(reviewWords.Count, maxLength); j++)
            {
                int wordIndex = wordList.IndexOf(reviewWords[j]);
                x[j * wordList.Count + wordIndex] = 1;
            }

            return x;
        }
    }
}