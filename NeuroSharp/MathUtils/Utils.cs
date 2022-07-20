using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using System.Linq;
using System.Drawing.Imaging;
using System.Drawing;

namespace NeuroSharp.MathUtils
{
    public static class Utils
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
        public static Matrix<double> Unflatten(Vector<double> vec)
        {
            int dim = (int)Math.Round(Math.Sqrt(vec.Count));
            Matrix<double> mtx = Matrix<double>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mtx[j, i] = vec[i * dim + j];
            return mtx;
        }

        public static double GetInitialWeight(int layerInputSize)
        {
            double sigma = Math.Sqrt(2f/layerInputSize);
            return (double)new Normal(0, sigma).Sample();
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

        public static void ToImage(Matrix<double> img)
        {
            int width = img.ColumnCount;
            int height = img.RowCount;

            Bitmap bitmap = new Bitmap(width, height);
            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    Color c = Color.FromArgb((byte)(img[x, y] * 255), (byte)(img[x, y] * 255), (byte)(img[x, y] * 255), 255);
                    bitmap.SetPixel(x, y, c);
                }
            }
            bitmap.Save(@"C:\Users\Isaac\Desktop\test.bmp");
        }
    }
}
