using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Optimizers
{
    public class Adam
    {
        private Matrix<double> _meanWeightGradient;
        private Vector<double> _meanBiasGradient;

        private Matrix<double> _varianceWeightGradient;
        private Vector<double> _varianceBiasGradient;

        private Matrix<double> _meanWeightGradCorrection;
        private Vector<double> _meanBiasGradCorrection;

        private Matrix<double> _varianceWeightGradCorrection;
        private Vector<double> _varianceBiasGradCorrection;

        public Adam(int inputSize, int outputSize)
        {
            _meanWeightGradient = Matrix<double>.Build.Dense(inputSize, outputSize);
            _meanBiasGradient = Vector<double>.Build.Dense(outputSize);
            _varianceWeightGradient = Matrix<double>.Build.Dense(inputSize, outputSize);
            _varianceBiasGradient = Vector<double>.Build.Dense(outputSize);
        }

        public void Step(ParameterizedLayer layer, int t, double eta = 0.001f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 0.0000001f, bool includeBias = true)
        {
            _meanWeightGradient = beta1 * _meanWeightGradient + (1 - beta1) * layer.WeightGradient;
            if(includeBias)
                _meanBiasGradient = beta1 * _meanBiasGradient + (1 - beta1) * layer.BiasGradient;

            _varianceWeightGradient = beta2 * _varianceWeightGradient + (1 - beta2) * layer.WeightGradient.PointwisePower(2);
            if(includeBias)
                _varianceBiasGradient = beta2 * _varianceBiasGradient + (1 - beta2) * layer.BiasGradient.PointwisePower(2);

            _meanWeightGradCorrection = _meanWeightGradient / (1 - Math.Pow(beta1, t));           
            _varianceWeightGradCorrection = _varianceWeightGradient / (1 - Math.Pow(beta2, t));
            layer.Weights -= eta * (_meanWeightGradCorrection.PointwiseDivide(_varianceWeightGradCorrection.PointwiseSqrt() + Matrix<double>.Build.One * epsilon));

            if(includeBias)
            {
                _meanBiasGradCorrection = _meanBiasGradient / (1 - Math.Pow(beta1, t));
                _varianceBiasGradCorrection = _varianceBiasGradient / (1 - Math.Pow(beta2, t));
                layer.Bias -= eta * (_meanBiasGradCorrection.PointwiseDivide(_varianceBiasGradCorrection.PointwiseSqrt() + Vector<double>.Build.One * epsilon));
            }
        }
    }

    public class AdamOutput
    {
        public Matrix<double> Weights { get; set; }
        public Vector<double> Bias { get; set; }
        public Matrix<double> MeanWeightGradient { get; set; }
        public Vector<double> MeanBiasGradient { get; set; }
        public Matrix<double> VarianceWeightGradient { get; set; }
        public Vector<double> VarianceBiasGradient { get; set; }
    }
}
