using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics;

namespace NeuroSharp.Optimizers
{
    public class Adam
    {
        private Matrix<double>[] _meanWeightGradient;
        private Vector<double> _meanBiasGradient;

        private Matrix<double>[] _varianceWeightGradient;
        private Vector<double> _varianceBiasGradient;

        private Matrix<double>[] _meanWeightGradCorrection;
        private Vector<double> _meanBiasGradCorrection;

        private Matrix<double>[] _varianceWeightGradCorrection;
        private Vector<double> _varianceBiasGradCorrection;

        public Adam(int inputSize, int outputSize, int weightCount = 1)
        {
            _meanWeightGradient = new Matrix<double>[weightCount];
            for (int i = 0; i < weightCount; i++)
                _meanWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            _meanBiasGradient = Vector<double>.Build.Dense(outputSize);
            _varianceWeightGradient = new Matrix<double>[weightCount];
            for (int i = 0; i < weightCount; i++)
                _varianceWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            _varianceBiasGradient = Vector<double>.Build.Dense(outputSize);
            _meanWeightGradCorrection = new Matrix<double>[weightCount];
            _varianceWeightGradCorrection = new Matrix<double>[weightCount];
        }

        public void Step(ParameterizedLayer layer, int t, double eta = 0.001f, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 0.0000001f, bool includeBias = true)
        {
            //for(int i = 0; i < layer.Weights.Length; i++)
            Parallel.For(0, layer.Weights.Length, i =>
            {
                _meanWeightGradient[i] = beta1 * _meanWeightGradient[i] + (1 - beta1) * layer.WeightGradient[i];
                _varianceWeightGradient[i] = beta2 * _varianceWeightGradient[i] + (1 - beta2) * layer.WeightGradient[i].PointwisePower(2);
                _meanWeightGradCorrection[i] = _meanWeightGradient[i] / (1 - Math.Pow(beta1, t));
                _varianceWeightGradCorrection[i] = _varianceWeightGradient[i] / (1 - Math.Pow(beta2, t));
                layer.Weights[i] -= eta * (_meanWeightGradCorrection[i].PointwiseDivide(_varianceWeightGradCorrection[i].PointwiseSqrt() + epsilon));
            });
            if (includeBias)
            {
                _meanBiasGradient = beta1 * _meanBiasGradient + (1 - beta1) * layer.BiasGradient;
                _varianceBiasGradient = beta2 * _varianceBiasGradient + (1 - beta2) * layer.BiasGradient.PointwisePower(2);
                _meanBiasGradCorrection = _meanBiasGradient / (1 - Math.Pow(beta1, t));
                _varianceBiasGradCorrection = _varianceBiasGradient / (1 - Math.Pow(beta2, t));
                layer.Bias -= eta * (_meanBiasGradCorrection.PointwiseDivide(_varianceBiasGradCorrection.PointwiseSqrt() + Vector<double>.Build.One * epsilon));
            }
        }
    }
}
