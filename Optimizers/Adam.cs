using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Optimizers
{
    public class Adam
    {
        private Matrix<float> _meanWeightGradient;
        private Vector<float> _meanBiasGradient;

        private Matrix<float> _varianceWeightGradient;
        private Vector<float> _varianceBiasGradient;

        private Matrix<float> _meanWeightGradCorrection;
        private Vector<float> _meanBiasGradCorrection;

        private Matrix<float> _varianceWeightGradCorrection;
        private Vector<float> _varianceBiasGradCorrection;

        public Adam(int inputSize, int outputSize)
        {
            _meanWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            _meanBiasGradient = Vector<float>.Build.Dense(outputSize);
            _varianceWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            _varianceBiasGradient = Vector<float>.Build.Dense(outputSize);
        }

        public void Step(ParameterizedLayer layer, int t, float eta = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 0.0000001f, bool includeBias = true)
        {
            _meanWeightGradient = beta1 * _meanWeightGradient + (1 - beta1) * layer.WeightGradient;
            if(includeBias)
                _meanBiasGradient = beta1 * _meanBiasGradient + (1 - beta1) * layer.BiasGradient;

            _varianceWeightGradient = beta2 * _varianceWeightGradient + (1 - beta2) * layer.WeightGradient.PointwisePower(2);
            if(includeBias)
                _varianceBiasGradient = beta2 * _varianceBiasGradient + (1 - beta2) * layer.BiasGradient.PointwisePower(2);

            _meanWeightGradCorrection = _meanWeightGradient / (1 - MathF.Pow(beta1, t));           
            _varianceWeightGradCorrection = _varianceWeightGradient / (1 - MathF.Pow(beta2, t));
            layer.Weights -= eta * (_meanWeightGradCorrection.PointwiseDivide(_varianceWeightGradCorrection.PointwiseSqrt() + Matrix<float>.Build.One * epsilon));

            if(includeBias)
            {
                _meanBiasGradCorrection = _meanBiasGradient / (1 - MathF.Pow(beta1, t));
                _varianceBiasGradCorrection = _varianceBiasGradient / (1 - MathF.Pow(beta2, t));
                layer.Bias -= eta * (_meanBiasGradCorrection.PointwiseDivide(_varianceBiasGradCorrection.PointwiseSqrt() + Vector<float>.Build.One * epsilon));
            }
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
