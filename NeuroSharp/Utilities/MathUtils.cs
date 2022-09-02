using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Linq;
using System.Drawing.Imaging;
using System.Drawing;

namespace NeuroSharp.Utilities
{
    public static class MathUtils
    {
        public static double Nextdouble(double min, double max)
        {
            System.Random random = new System.Random();
            double val = (double)(random.NextDouble() * (max - min) + min);
            return (double)val;
        }
        public static Vector<double> Flatten(Matrix<double> mtx)
        {
            List<double> f = new List<double>();
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    f.Add(mtx[i, j]);
            return Vector<double>.Build.DenseOfArray(f.ToArray());
        }

        public static Vector<double> Flatten(Vector<double>[] mtx) //todo: write test for this
        {
            List<double> f = new List<double>();
            for (int i = 0; i < mtx.Length; i++)
                for (int j = 0; j < mtx[i].Count; j++)
                    f.Add(mtx[i][j]);
            return Vector<double>.Build.DenseOfArray(f.ToArray());
        }
        public static Matrix<double> Unflatten(Vector<double> vec)
        {
            int dim = (int)Math.Round(Math.Sqrt(vec.Count));
            Matrix<double> mtx = Matrix<double>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mtx[j, i] = vec[i * dim + j];
            return mtx;
        }

        public static Vector<double>[] UnflattenVecArray(Vector<double> vec, int w, int h) //todo: write test for this
        {
            Vector<double>[] output = new Vector<double>[w];
            for (int i = 0; i < w; i++)
            {
                Vector<double> col = Vector<double>.Build.Dense(h);
                for (int j = 0; j < h; j++)
                    col[j] = vec[i * w + j];
                output[i] = col;
            }

            return output;
        }
        public static Matrix<double> Unflatten(Vector<double> vec, int rowCount, int colCount)
        {
            Matrix<double> mtx = Matrix<double>.Build.Dense(rowCount, colCount);
            for (int j = 0; j < colCount; j++)
                for (int i = 0; i < rowCount; i++)
                    mtx[i, j] = vec[j * rowCount + i];
            return mtx;
        }
        public static double GetInitialWeightFromInputSize(int layerInputSize)
        {
            double sigma = Math.Sqrt(2f/layerInputSize);
            return new Normal(0, sigma).Sample();
        }

        public static double GetInitialWeightFromRange(double lowerBound, double upperBound)
        {
            Random rand = new Random();
            return rand.NextDouble() * (upperBound - lowerBound) + lowerBound;
        }

        public static Vector<double> FiniteDifferencesGradient(Func<Vector<double>, Vector<double>> f, Vector<double> x, double h = 0.000001f)
        {
            Vector<double> grad = Vector<double>.Build.Dense(x.Count);
            Vector<double> hvec = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                hvec[i] = h;
                Vector<double> diffQuot = (f(x + hvec) - f(x - hvec)) / (2 * h);
                grad[i] = diffQuot.FirstOrDefault(t => t != 0);
                hvec = Vector<double>.Build.Dense(x.Count);
            }

            return grad;
        }

        public static Vector<double> FiniteDifferencesGradient(Func<Vector<double>, double> f, Vector<double> x, double h = 0.000001f)
        {
            Vector<double> grad = Vector<double>.Build.Dense(x.Count);
            Vector<double> hvec = Vector<double>.Build.Dense(x.Count);
            for (int i = 0; i < x.Count; i++)
            {
                hvec[i] = h;
                grad[i] = (f(x + hvec) - f(x - hvec)) / (2 * h);
                hvec = Vector<double>.Build.Dense(x.Count);
            }

            return grad;
        }
    }
}
