using MathNet.Numerics.LinearAlgebra;
using System.Linq;
using System;

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

        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double SigmoidPrime(double x)
        {
            double expNegX = Math.Exp(-x);
            return expNegX / Math.Pow(1 + expNegX, 2);
        }

        public static Vector<double> Softmax(Vector<double> x)
        {
            Vector<double> result = Vector<double>.Build.Dense(x.Count);
            double expSum = x.Sum(f => Math.Exp(f));
            for(int i = 0; i < x.Count; i++)
                result[i] = Math.Exp(x[i])/expSum;
            return result;
        }

        public static Matrix<double> SoftmaxPrime(Vector<double> x)
        {
            Matrix<double> result = Matrix<double>.Build.Dense(x.Count, x.Count);
            double expSum = x.Sum(f => Math.Exp(f));
            for (int i = 0; i < x.Count; i++)
            {
                for (int j = 0; j < x.Count; j++)
                { 
                    if(i == j)
                        result[i, j] = Math.Exp(x[i]) * (expSum - Math.Exp(x[i]));
                    else
                        result[i, j] = -Math.Exp(x[i] + x[j]);
                }
                    
            }
            return result / Math.Pow(expSum, 2);
        }
    }
}
