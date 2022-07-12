using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Statistics;

namespace NeuroSharp.MathUtils
{
    public static class LossFunctions
    {
        public static float MeanSquaredError(Vector<float> truth, Vector<float> test)
        {
            return (float)Statistics.Mean((truth - test).PointwisePower(2));
        }

        public static Vector<float> MeanSquaredErrorPrime(Vector<float> truth, Vector<float> test)
        {
            return 2 * (test - truth) / truth.Count();
        }

        public static float CategoricalCrossentropy(Vector<float> truth, Vector<float> test)
        {
            float sum = 0;
            for(int i = 0; i < truth.Count; i++)
                sum += truth[i]*MathF.Log10(test[i]);
            return -sum;
        }

        public static Vector<float> CategoricalCrossentropyPrime(Vector<float> truth, Vector<float> test)
        {
            return test - truth;
        }
    }
}
