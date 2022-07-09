using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Factorization;
using MathNet.Numerics.Statistics;
using System.Linq;
using MathNet.Numerics.LinearAlgebra.Single;

namespace NeuroSharp.MathUtils
{
    public static class PCA
    {
        public static Vector<float> GetPrincipleComponents(Vector<float> input, int componentCount)
        {
            int dim = (int)Math.Round(Math.Sqrt(input.Count));
            Matrix<float> mtx = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mtx[i, j] = input[dim * i + j];

            // columnwise mean
            Vector<float> colMean = mtx.ColumnSums() / mtx.ColumnCount;

            // normalize using columnwise mean
            var ones = Vector<float>.Build.Dense(colMean.Count);
            for (int i = 0; i < colMean.Count; i++)
                ones[i] = 1;
            mtx -= ones * colMean;

            // compute covariance and truncate it's rows with smaller eigenvaleus
            Matrix<float> covariance = GetCovarianceMatrix(mtx);
            Matrix<float> featureMtx = EigenSort(covariance, componentCount);

            //take inner product of each feature matrix column (original images (mtx)) with the principle vectors (columns of featureMtx)
            // this is just transpose(featureMtx) * cenered original images (mtx)

            List<float> features = new List<float>();
            for (int i = 0; i < featureMtx.RowCount; i++)
                for(int j = 0; j < featureMtx.ColumnCount; j++)
                    features.Add(featureMtx[i, j]);

            return Vector<float>.Build.DenseOfArray(features.ToArray());
        }

        static Matrix<float> EigenSort(Matrix<float> mtx, int componentCount)
        {
            List<(float, Vector<float>)> eigValsVecs = new List<(float, Vector<float>)>();

            Evd<float> evdDecomp = mtx.Evd();
            for(int i = 0; i < evdDecomp.EigenValues.Count; i++)
            {
                eigValsVecs.Add(((float)evdDecomp.EigenValues[i].Real, Vector<float>.Build.DenseOfEnumerable(evdDecomp.EigenVectors.Column(i))));
            }
            
            eigValsVecs = eigValsVecs.OrderByDescending(x => x.Item1).Take(componentCount).ToList();
            Matrix<float> eigenColumns = Matrix<float>.Build.Dense(eigValsVecs.Count, eigValsVecs[0].Item2.Count);
            for(int i = 0; i < eigValsVecs.Count; i++)
            {
                for(int j = 0; j < eigValsVecs[0].Item2.Count; j++)
                {
                    eigenColumns[i, j] = eigValsVecs[i].Item2[j];
                }
            }

            return eigenColumns;
        }

        //https://stackoverflow.com/questions/32256998/find-covariance-of-math-net-matrix
        static Matrix<float> GetCovarianceMatrix(Matrix<float> mtx)
        {
            var columnAverages = mtx.ColumnSums() / mtx.RowCount;
            var centeredColumns = mtx.EnumerateColumns().Zip(columnAverages, (col, avg) => col - avg);
            var centered = DenseMatrix.OfColumnVectors(centeredColumns);
            var normalizationFactor = mtx.RowCount == 1 ? 1 : mtx.RowCount - 1;
            return centered.TransposeThisAndMultiply(centered) / normalizationFactor;
        }
    }
}
