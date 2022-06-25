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
            float ret = (float)Statistics.Mean((truth - test).PointwisePower(2));
            if(float.IsNaN(ret))
            {

            }
            return ret;
        }

        //returns ∂E/∂Y
        public static Vector<float> MeanSquaredErrorPrime(Vector<float> truth, Vector<float> test)
        {
            return 2 * (test - truth) / truth.Count();
        }
    }
}
