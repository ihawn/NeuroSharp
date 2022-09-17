using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Data
{
    public static class DataUtils
    {
        public static Vector<double> OneHotEncodeVector(Vector<double> x, int categoryCount)
        {
            Vector<double> output = Vector<double>.Build.Dense(x.Count * categoryCount);

            for (int i = 0; i < x.Count; i++)
                output[i * categoryCount + (int)Math.Round(x[i])] = 1;

            return output;
        }
    }
}