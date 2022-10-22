using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Linq;
using System.Drawing.Imaging;
using System.Drawing;

namespace NeuroSharp.Utilities
{
    public static class MathUtils
    {
        public static Vector<double> Flatten(Matrix<double> mtx)
        {
            return Vector<double>.Build.DenseOfArray(mtx.ToRowMajorArray());
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
            return Matrix<double>.Build.DenseOfColumnMajor(dim, dim, vec);
        }
        public static Matrix<double> Unflatten(Vector<double> vec, int rowCount, int colCount)
        {
            //todo: make sure row/column relationship here is correct
            return Matrix<double>.Build.DenseOfColumnMajor(rowCount, colCount, vec);
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
        public static Vector<double> FiniteDifferencesGradient(Func<Vector<double>, double> f, Vector<double> x, double h = 0.000001)
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
        public static Matrix<double> TransposianShift(Matrix<double> mtx)
        {
            List<double> f = new List<double>();
            for (int j = 0; j < mtx.ColumnCount; j++)
                for (int i = 0; i < mtx.RowCount; i++)
                    f.Add(mtx[i, j]);
            return Unflatten(Vector<double>.Build.DenseOfEnumerable(f), mtx.ColumnCount, mtx.RowCount);
        }
        public static Matrix<double> TransposianAdd(Matrix<double> mtx1, Matrix<double> mtx2)
        {
            return mtx1.Add(Matrix<double>.Build.DenseOfRowMajor(
                mtx2.RowCount, 
                mtx2.ColumnCount, 
                mtx2.ToColumnMajorArray()
            ));
        }
        public static int GetConvolutionOutputSize(int inputSize, int kernel, int stride)
        {
            return (int)Math.Floor((Math.Round(Math.Sqrt(inputSize)) - kernel) / stride + 1);
        }
    }
}
