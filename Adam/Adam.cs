using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

namespace Adam
{
    public class Adam
    {
        public float Eta { get { return 0.01f; } }
        public float Beta1 { get { return 0.9f; } }
        public float Beta2 { get { return 0.999f; } }
        public float Epsilon { get { return 0.00000001f; } }
        public Matrix<float> MeanWeightGradient { get; set; }
        public Vector<float> MeanBiasGradient { get; set; }
        public Matrix<float> VarianceWeightGradient { get; set; }
        public Vector<float> VarianceBiasGradient { get; set; }
        public Matrix<float> Weight { get; set; }
        public Vector<float> Bias { get; set; }
        public Matrix<float> WeightGradient { get; set; }
        public Vector<float> BiasGradient { get; set; }

        public Adam(Matrix<float> weight)
        {
            Weight = weight;

            MeanWeightGradient = Matrix<float>.Build.Dense(1, 2);
            VarianceWeightGradient = Matrix<float>.Build.Dense(1, 2);
        }

        public Adam(Matrix<float> weight, Vector<float> bias, Matrix<float> weightGradient, Vector<float> biasGradient)
        {
            int i = weightGradient.RowCount;
            int j = weightGradient.ColumnCount;
            int k = biasGradient.Count;
            MeanWeightGradient = Matrix<float>.Build.Dense(i, j);
            MeanBiasGradient = Vector<float>.Build.Dense(k);
            VarianceWeightGradient = Matrix<float>.Build.Dense(i, j);
            VarianceBiasGradient = Vector<float>.Build.Dense(k);
            Weight = weight;
            Bias = bias;
            WeightGradient = weightGradient;
            BiasGradient = biasGradient;
        }

        public void StepForX2Y2(int maxIterations = 10000)
        {
            float f(float x, float y) { return x * x + y * y; }
            Matrix<float> df(float x, float y)
            {
                float[,] f = new float[,]
                {
                    { 2*x, 2*y }
                };
                return Matrix<float>.Build.DenseOfArray(f);
            }

            Matrix<float> oldWeight = Matrix<float>.Build.Dense(Weight.RowCount, Weight.ColumnCount);

            for (int t = 1; t <= maxIterations; t++)
            {
                MeanWeightGradient = Beta1 * MeanWeightGradient + (1 - Beta1) * df(Weight[0,0], Weight[0,1]);

                VarianceWeightGradient = Beta2 * VarianceWeightGradient + (1 - Beta2) * df(Weight[0, 0], Weight[0, 1]).PointwisePower(2);


                Matrix<float> meanWeightGradCorrection = MeanWeightGradient / (1 - MathF.Pow(Beta1, t));
                Matrix<float> varianceWeightGradCorrection = VarianceWeightGradient / (1 - MathF.Pow(Beta2, t));


                Weight -= Eta * (meanWeightGradCorrection.PointwiseDivide(varianceWeightGradCorrection.PointwiseSqrt() + Matrix<float>.Build.One * Epsilon));

                if (t > 1 && (Weight - oldWeight).RowSums().L2Norm() < 0.0001f)
                    break;

                oldWeight = Weight;
            }
        }

        public void Step(int maxIterations = 100)
        {
            Matrix<float> oldWeight = Matrix<float>.Build.Dense(Weight.RowCount, Weight.ColumnCount);

            for(int t = 1; t <= maxIterations; t++)
            {
                MeanWeightGradient = Beta1 * MeanWeightGradient + (1 - Beta1) * WeightGradient;
                MeanBiasGradient = Beta1 * MeanBiasGradient + (1 - Beta1) * BiasGradient;

                VarianceWeightGradient = Beta2 * VarianceWeightGradient + (1 - Beta2) * WeightGradient.PointwisePower(2);
                VarianceBiasGradient = Beta2 * VarianceBiasGradient + (1 - Beta2) * BiasGradient;


                Matrix<float> meanWeightGradCorrection = MeanWeightGradient / (1 - MathF.Pow(Beta1, t));
                Vector<float> meanBiasGradCorrection = MeanBiasGradient / (1 - MathF.Pow(Beta1, t));
                Matrix<float> varianceWeightGradCorrection = VarianceWeightGradient / (1 - MathF.Pow(Beta2, t));
                Vector<float> varianceBiasGradCorrection = VarianceBiasGradient / (1 - MathF.Pow(Beta2, t));


                Weight -= Eta * (meanWeightGradCorrection.PointwiseDivide(varianceWeightGradCorrection.PointwiseSqrt() + Matrix<float>.Build.One * Epsilon));
                Bias -= Eta * (meanBiasGradCorrection.PointwiseDivide(varianceBiasGradCorrection.PointwiseSqrt() + Vector<float>.Build.One * Epsilon));

                if (t > 1 && (Weight - oldWeight).RowSums().L2Norm() < 0.0001f)
                    break;

                oldWeight = Weight;
            }
        }
    }
}
