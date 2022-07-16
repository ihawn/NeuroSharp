using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Linq;

namespace NeuroSharp.MathUtils
{
    public static class Utils
    {
        public static float Nextfloat(float min, float max)
        {
            System.Random random = new System.Random();
            float val = (float)(random.NextDouble() * (max - min) + min);
            return (float)val;
        }
        public static Vector<float> Flatten(Matrix<float> mtx)
        {
            List<float> f = new List<float>();
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    f.Add(mtx[i, j]);
            return Vector<float>.Build.DenseOfArray(f.ToArray());
        }
        public static Matrix<float> Unflatten(Vector<float> vec)
        {
            int dim = (int)Math.Round(Math.Sqrt(vec.Count));
            Matrix<float> mtx = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mtx[j, i] = vec[i * dim + j];
            return mtx;
        }

        public static float GetInitialWeight(int layerInputSize)
        {
            float sigma = MathF.Sqrt(2f/layerInputSize);
            return (float)new Normal(0, sigma).Sample();
        }

        public static Vector<double> FiniteDifferencesGradient(Func<Vector<double>, Vector<double>> f, Vector<double> x, double h = 0.000001f)
        {
            Vector<double> hvec = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
                hvec[i] = h;
            return (f(x + hvec) - f(x - hvec)) / (2 * hvec);
        }
    }
}
