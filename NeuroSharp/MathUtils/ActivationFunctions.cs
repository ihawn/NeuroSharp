using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace NeuroSharp
{
    public static class ActivationFunctions
    {
        public static float Tanh(float x)
        {
            return MathF.Tanh(x);
        }

        public static float TanhPrime(float x)
        {
            return 1 - MathF.Pow(MathF.Tanh(x), 2);
        }

        public static float Relu(float x)
        {
            return MathF.Max(0, x);
        }

        public static float ReluPrime(float x)
        {
            return x <= 0 ? 0 : 1;
        }

        public static Vector<float> Softmax(Vector<float> x)
        {
            Vector<float> result = Vector<float>.Build.Dense(x.Count);
            float expSum = x.Sum(f => MathF.Exp(f));
            for(int i = 0; i < x.Count; i++)
                result[i] = MathF.Exp(x[i])/expSum;
            return result;
        }

        public static float PointwiseSoftmax(float x, float expSum)
        {
            return MathF.Exp(x) / expSum;
        }

        public static Matrix<float> SoftmaxPrime(Vector<float> x)
        {
            Matrix<float> result = Matrix<float>.Build.Dense(x.Count, x.Count);
            float expSum = x.Sum(f => MathF.Exp(f));
            for (int i = 0; i < x.Count; i++)
                for(int j = 0; j < x.Count; j++)
                    result[i, j] = i == j ? PointwiseSoftmax(x[i], expSum) * (1 - PointwiseSoftmax(x[j], expSum)) : 
                                           -PointwiseSoftmax(x[j], expSum) * PointwiseSoftmax(x[i], expSum);
            return result;
        }
    }
}
