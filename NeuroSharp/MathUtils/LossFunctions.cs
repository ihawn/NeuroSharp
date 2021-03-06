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
        public static double MeanSquaredError(Vector<double> truth, Vector<double> test)
        {
            return (double)Statistics.Mean((truth - test).PointwisePower(2));
        }

        public static Vector<double> MeanSquaredErrorPrime(Vector<double> truth, Vector<double> test)
        {
            return 2 * (test - truth) / truth.Count();
        }

        public static double CategoricalCrossentropy(Vector<double> truth, Vector<double> test)
        {
            double sum = 0;
            for(int i = 0; i < truth.Count; i++)
                sum += truth[i]*Math.Log(test[i]);
            return -sum;
        }

        public static Vector<double> CategoricalCrossentropyPrime(Vector<double> truth, Vector<double> test)
        {
            //return test - truth;
            Vector<double> gradient = Vector<double>.Build.Dense(test.Count);
            for (int i = 0; i < gradient.Count; i++)
                gradient[i] = -truth[i] / test[i];
            return gradient;
        }
    }
}
