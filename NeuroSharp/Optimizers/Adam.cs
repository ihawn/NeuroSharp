using MathNet.Numerics.LinearAlgebra;
using Newtonsoft.Json;
using System;

namespace NeuroSharp.Optimizers
{
    public class Adam
    {
        [JsonProperty]
        private Matrix<double>[] _meanWeightGradient;
        [JsonProperty]
        private Vector<double>[] _meanBiasGradient;

        [JsonProperty]
        private Matrix<double>[] _varianceWeightGradient;
        [JsonProperty]
        private Vector<double>[] _varianceBiasGradient;

        [JsonProperty]
        private double _beta1;
        [JsonProperty]
        private double _beta2;
        [JsonProperty]
        private double _oneMinusBeta1;
        [JsonProperty]
        private double _oneMinusBeta2;
        [JsonProperty]
        private double _epsilon;

        public Adam(int inputSize, int outputSize, int weightCount = 1, int biasCount = 1, double beta1 = 0.9f, double beta2 = 0.999f, double epsilon = 0.0000001f)
        {
            _beta1 = beta1;
            _beta2 = beta2;
            _oneMinusBeta1 = 1 - beta1;
            _oneMinusBeta2 = 1 - beta2;
            _epsilon = epsilon;

            _meanWeightGradient = new Matrix<double>[weightCount];
            _meanBiasGradient = new Vector<double>[biasCount];
            for (int i = 0; i < weightCount; i++)
                _meanWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            for(int i = 0; i < biasCount; i++)
                _meanBiasGradient[i] = Vector<double>.Build.Dense(outputSize);
            
            _varianceWeightGradient = new Matrix<double>[weightCount];
            _varianceBiasGradient = new Vector<double>[biasCount];
            for (int i = 0; i < weightCount; i++)
                _varianceWeightGradient[i] = Matrix<double>.Build.Dense(inputSize, outputSize);
            for(int i = 0; i < biasCount; i++)
                _varianceBiasGradient[i] = Vector<double>.Build.Dense(outputSize);
        }
        
        //json constructor
        public Adam(Matrix<double>[] meanWeightGradient, Vector<double>[] meanBiasGradient, 
            Matrix<double>[] varianceWeightGradient, Vector<double>[] varianceBiasGradient, 
            double beta1, double beta2, double epsilon)
        {
            _meanWeightGradient = meanWeightGradient;
            _meanBiasGradient = meanBiasGradient;
            _varianceWeightGradient = varianceWeightGradient;
            _varianceBiasGradient = varianceBiasGradient;
            _beta1 = beta1;
            _beta2 = beta2;
            _epsilon = epsilon;
        }

        public void Step(ParameterizedLayer layer, int t, double eta = 0.001f, bool includeBias = true)
        {
            for (int i = 0; i < layer.Weights.Length; i++)
                UpdateWeightParameters(layer, t, eta, i);
            if(includeBias)
                for(int i = 0; i < layer.Biases.Length; i++)
                    UpdateBiasParameters(layer, t, eta, i);
        }

        private void UpdateWeightParameters(ParameterizedLayer layer, int t, double eta, int i)
        {
            if (_meanWeightGradient[i].RowCount != layer.WeightGradients[i].RowCount ||
                _meanWeightGradient[i].ColumnCount != layer.WeightGradients[i].ColumnCount)
            {
                _meanWeightGradient[i] = Matrix<double>.Build.Dense(layer.WeightGradients[i].RowCount,
                    layer.WeightGradients[i].ColumnCount);
            }
            if (_varianceWeightGradient[i].RowCount != layer.WeightGradients[i].RowCount ||
                _varianceWeightGradient[i].ColumnCount != layer.WeightGradients[i].ColumnCount)
            {
                _varianceWeightGradient[i] = Matrix<double>.Build.Dense(layer.WeightGradients[i].RowCount,
                    layer.WeightGradients[i].ColumnCount);
            }
            
            _meanWeightGradient[i] = _meanWeightGradient[i].Multiply(_beta1).Add(layer.WeightGradients[i].Multiply(_oneMinusBeta1));
            _varianceWeightGradient[i] = _varianceWeightGradient[i].Multiply(_beta2).Add(layer.WeightGradients[i].PointwisePower(2).Multiply(_oneMinusBeta2));
            layer.Weights[i] -= _meanWeightGradient[i].Divide(1 - Math.Pow(_beta1, t)).PointwiseDivide(_varianceWeightGradient[i].Divide(1 - Math.Pow(_beta2, t)).PointwiseSqrt() + _epsilon).Multiply(eta);
        }

        private void UpdateBiasParameters(ParameterizedLayer layer, int t, double eta, int i)
        {
            if (_meanBiasGradient[i].Count != layer.BiasGradients[i].Count)
                _meanBiasGradient[i] = Vector<double>.Build.Dense(layer.BiasGradients[i].Count);
            if (_varianceBiasGradient[i].Count != layer.BiasGradients[i].Count)
                _varianceBiasGradient[i] = Vector<double>.Build.Dense(layer.BiasGradients[i].Count);

            _meanBiasGradient[i] = _meanBiasGradient[i].Multiply(_beta1).Add(layer.BiasGradients[i].Multiply(_oneMinusBeta1));
            _varianceBiasGradient[i] = _varianceBiasGradient[i].Multiply(_beta2).Add(layer.BiasGradients[i].PointwisePower(2).Multiply(_oneMinusBeta2));
            layer.Biases[i] -= _meanBiasGradient[i].Divide(1 - Math.Pow(_beta1, t)).PointwiseDivide(_varianceBiasGradient[i].Divide(1 - Math.Pow(_beta2, t)).PointwiseSqrt() + _epsilon).Multiply(eta);
        }
    }
}
