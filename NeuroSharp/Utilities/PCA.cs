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

namespace NeuroSharp.Utilities
{
    public static class PCA
    {
        public static Vector<double> GetPrincipleComponents(Vector<double> input, int componentCount)
        {
            int dim = (int)Math.Round(Math.Sqrt(input.Count));
            Matrix<double> mtx = Matrix<double>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    mtx[i, j] = input[dim * i + j];

            // columnwise mean
            Vector<double> colMean = mtx.ColumnSums() / mtx.ColumnCount;

            // normalize using columnwise mean
            var ones = Vector<double>.Build.Dense(colMean.Count);
            for (int i = 0; i < colMean.Count; i++)
                ones[i] = 1;
            mtx -= ones * colMean;

            // compute covariance and truncate it's rows with smaller eigenvaleus
            Matrix<double> covariance = Matrix<double>.Build.Dense(1, 1);// GetCovarianceMatrix(mtx);
            Matrix<double> featureMtx = EigenSort(covariance, componentCount);

            //take inner product of each feature matrix column (original images (mtx)) with the principle vectors (columns of featureMtx)
            // this is just transpose(featureMtx) * cenered original images (mtx)

            List<double> features = new List<double>();
            for (int i = 0; i < featureMtx.RowCount; i++)
                for(int j = 0; j < featureMtx.ColumnCount; j++)
                    features.Add(featureMtx[i, j]);

            return Vector<double>.Build.DenseOfArray(features.ToArray());
        }

        static Matrix<double> EigenSort(Matrix<double> mtx, int componentCount)
        {
            List<(double, Vector<double>)> eigValsVecs = new List<(double, Vector<double>)>();

            Evd<double> evdDecomp = mtx.Evd();
            for(int i = 0; i < evdDecomp.EigenValues.Count; i++)
            {
                eigValsVecs.Add(((double)evdDecomp.EigenValues[i].Real, Vector<double>.Build.DenseOfEnumerable(evdDecomp.EigenVectors.Column(i))));
            }
            
            eigValsVecs = eigValsVecs.OrderByDescending(x => x.Item1).Take(componentCount).ToList();
            Matrix<double> eigenColumns = Matrix<double>.Build.Dense(eigValsVecs.Count, eigValsVecs[0].Item2.Count);
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

        //need to figure how to get this to work with double precision arithmetic
        /*static Matrix<double> GetCovarianceMatrix(Matrix<double> mtx)
        {
            var columnAverages = mtx.ColumnSums() / mtx.RowCount;
            var centeredColumns = mtx.EnumerateColumns().Zip(columnAverages, (col, avg) => col - avg);
            var centered = DenseMatrix.OfColumnVectors(centeredColumns);
            var normalizationFactor = mtx.RowCount == 1 ? 1 : mtx.RowCount - 1;
            return centered.TransposeThisAndMultiply(centered) / normalizationFactor;
        }*/

    }
}
