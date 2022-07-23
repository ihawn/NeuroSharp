using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace NeuroSharp
{
    public static class ActivationFunctions
    {
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        public static double TanhPrime(double x)
        {
            return 1 - Math.Pow(Math.Tanh(x), 2);
        }

        public static double Relu(double x)
        {
            return Math.Max(0, x);
        }

        public static double ReluPrime(double x)
        {
            return x <= 0 ? 0 : 1;
        }

        public static Vector<double> Softmax(Vector<double> x)
        {
            Vector<double> result = Vector<double>.Build.Dense(x.Count);
            double expSum = x.Sum(f => Math.Exp(f));
            for(int i = 0; i < x.Count; i++)
                result[i] = Math.Exp(x[i])/expSum;
            return result;
        }

        public static double PointwiseSoftmax(double x, double expSum)
        {
            return Math.Exp(x) / expSum;
        }

        public static Matrix<double> SoftmaxPrime(Vector<double> x)
        {
            Matrix<double> result = Matrix<double>.Build.Dense(x.Count, x.Count);
            /*double expSum = x.Sum(f => Math.Exp(f));
            for (int i = 0; i < x.Count; i++)
                for(int j = 0; j < x.Count; j++)
                    result[i, j] = i == j ? PointwiseSoftmax(x[i], expSum) * (1 - PointwiseSoftmax(x[j], expSum)) : 
                                           -PointwiseSoftmax(x[j], expSum) * PointwiseSoftmax(x[i], expSum);*/
            double expSum = x.Sum(f => Math.Exp(f));
            for (int i = 0; i < x.Count; i++)
            {
                for (int j = 0; j < x.Count; j++)
                { 
                    if(i == j)
                    {
                        result[i, j] = Math.Exp(x[i]) * (expSum - Math.Exp(x[i]));
                    }
                    else
                    {
                        result[i, j] = -Math.Exp(x[i] + x[j]);
                    }
                }
                    
            }
            return result / Math.Pow(expSum, 2);
        }
    }
}
