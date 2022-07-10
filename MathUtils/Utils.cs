using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.MathUtils
{
    public static class Utils
    {
        public static float NextFloat(float min, float max)
        {
            System.Random random = new System.Random();
            double val = (random.NextDouble() * (max - min) + min);
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
    }
}
