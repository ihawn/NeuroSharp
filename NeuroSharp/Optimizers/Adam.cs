using MathNet.Numerics.LinearAlgebra;
using System.Threading.Tasks;

namespace NeuroSharp.Optimizers
{
    public class Adam
    {
        private Matrix<double>[] _meanWeightGradient;
        private Vector<double> _meanBiasGradient;

        private Matrix<double>[] _varianceWeightGradient;
        private Vector<double> _varianceBiasGradient;

        private double _beta1;
        private double _beta2;
        private double _oneMinusBeta1;
        private double _oneMinusBeta2;
        private double _epsilon;

        public Adam(int inputSize, int outputSize, int weightCount = 1, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 0.0000001f)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _oneMinusBeta1 = 1 - beta1;
            _oneMinusBeta2 = 1 - beta2;
            _epsilon = epsilon;

            _meanWeightGradient = new Matrix<double>[weightCount];
            for (int i = 0; i < weightCount; i++)
                _meanWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            _meanBiasGradient = Vector<double>.Build.Dense(outputSize);
            _varianceWeightGradient = new Matrix<double>[weightCount];
            for (int i = 0; i < weightCount; i++)
                _varianceWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            _varianceBiasGradient = Vector<double>.Build.Dense(outputSize);
        }

        public void Step(ParameterizedLayer layer, int t, double eta = 0.001f, bool includeBias = true)
        {
            for (int i = 0; i < layer.Weights.Length; i++)
                UpdateWeightParameters(layer, t, eta, i);
            if (includeBias)
                UpdateBiasParameters(layer, t, eta);
        }

        private void UpdateWeightParameters(ParameterizedLayer layer, int t, double eta, int i)
        {
            _meanWeightGradient[i] = _meanWeightGradient[i].Multiply(_beta1).Add(layer.WeightGradients[i].Multiply(_oneMinusBeta1));
            _varianceWeightGradient[i] = _varianceWeightGradient[i].Multiply(_beta2).Add(layer.WeightGradients[i].PointwisePower(2).Multiply(_oneMinusBeta2));
            layer.Weights[i] -= _meanWeightGradient[i].Divide(1 - Math.Pow(_beta1, t)).PointwiseDivide(_varianceWeightGradient[i].Divide(1 - Math.Pow(_beta2, t)).PointwiseSqrt() + _epsilon).Multiply(eta);
        }

        private void UpdateBiasParameters(ParameterizedLayer layer, int t, double eta)
        {
            _meanBiasGradient = _meanBiasGradient.Multiply(_beta1).Add(layer.BiasGradient.Multiply(_oneMinusBeta1));
            _varianceBiasGradient = _varianceBiasGradient.Multiply(_beta2).Add(layer.BiasGradient.PointwisePower(2).Multiply(_oneMinusBeta2));
            layer.Bias -= _meanBiasGradient.Divide(1 - Math.Pow(_beta1, t)).PointwiseDivide(_varianceBiasGradient.Divide(1 - Math.Pow(_beta2, t)).PointwiseSqrt() + _epsilon).Multiply(eta);
        }
    }
}
