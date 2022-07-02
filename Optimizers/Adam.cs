using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Optimizers
{
    public static class Adam
    {
        public static AdamOutput Step(Matrix<float> weight, Vector<float> bias, Matrix<float> weightGradient, Vector<float> biasGradient, int t,
            Matrix<float> meanWeightGradient, Vector<float> meanBiasGradient, Matrix<float> varianceWeightGradient, Vector<float> varianceBiasGradient,
            float eta = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.0000001f)
        {
            int i = weightGradient.RowCount;
            int j = weightGradient.ColumnCount;

            meanWeightGradient = meanWeightGradient ?? Matrix<float>.Build.Dense(i, j);
            meanBiasGradient = meanBiasGradient ?? Vector<float>.Build.Dense(j);
            varianceWeightGradient = varianceWeightGradient ?? Matrix<float>.Build.Dense(i, j);
            varianceBiasGradient = varianceBiasGradient ?? Vector<float>.Build.Dense(j);
            
            //
            // Adam step
            //
            meanWeightGradient = beta1 * meanWeightGradient + (1 - beta1) * weightGradient;
            meanBiasGradient = beta1 * meanBiasGradient + (1 - beta1) * biasGradient;

            varianceWeightGradient = beta2 * varianceWeightGradient + (1 - beta2) * weightGradient.PointwisePower(2);
            varianceBiasGradient = beta2 * varianceBiasGradient + (1 - beta2) * biasGradient.PointwisePower(2);

            Matrix<float> meanWeightGradCorrection = meanWeightGradient / (1 - MathF.Pow(beta1, t));
            Vector<float> meanBiasGradCorrection = meanBiasGradient / (1 - MathF.Pow(beta1, t));
            Matrix<float> varianceWeightGradCorrection = varianceWeightGradient / (1 - MathF.Pow(beta2, t));
            Vector<float> varianceBiasGradCorrection = varianceBiasGradient / (1 - MathF.Pow(beta2, t));

            weight -= eta * (meanWeightGradCorrection.PointwiseDivide(varianceWeightGradCorrection.PointwiseSqrt() + Matrix<float>.Build.One * epsilon));
            bias -= eta * (meanBiasGradCorrection.PointwiseDivide(varianceBiasGradCorrection.PointwiseSqrt() + Vector<float>.Build.One * epsilon));

            return new AdamOutput()
            {
                Weights = weight,
                Bias = bias,
                MeanWeightGradient = meanWeightGradient,
                MeanBiasGradient = meanBiasGradient,
                VarianceWeightGradient = varianceWeightGradient,
                VarianceBiasGradient = varianceBiasGradient
            };

        }
    }

    public class AdamOutput
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }
        public Matrix<float> MeanWeightGradient { get; set; }
        public Vector<float> MeanBiasGradient { get; set; }
        public Matrix<float> VarianceWeightGradient { get; set; }
        public Vector<float> VarianceBiasGradient { get; set; }
    }
}
