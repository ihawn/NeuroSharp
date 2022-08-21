using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class FullyConnectedLayer : ParameterizedLayer
    {
        [JsonProperty]
        private Adam _adam;

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            LayerType = LayerType.FullyConnected;
            WeightGradients = new Matrix<double>[] { Matrix<double>.Build.Dense(inputSize, outputSize) };
            Weights = new Matrix<double>[] { Matrix<double>.Build.Random(inputSize, outputSize) };
            Bias = Vector<double>.Build.Random(outputSize);
            BiasGradient = Vector<double>.Build.Dense(outputSize);
            _adam = new Adam(inputSize, outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[0][i, j] = MathUtils.GetInitialWeight(inputSize);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = MathUtils.GetInitialWeight(inputSize);
        }
        
        //json constructor
        public FullyConnectedLayer(Matrix<double> weight, Vector<double> bias, Matrix<double> weightGradient,
            Vector<double> biasGradient, Adam adam, bool accumulateGradients)
        {
            LayerType = LayerType.FullyConnected;
            Weights = new Matrix<double>[] { weight };
            WeightGradients = new Matrix<double>[] { weightGradient };
            Bias = bias;
            BiasGradient = biasGradient;
            _adam = adam;
            SetGradientAccumulation(accumulateGradients);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Input * Weights[0] + Bias;
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            WeightGradients[0] = AccumulateGradients ? WeightGradients[0] + Input.OuterProduct(outputError) : Input.OuterProduct(outputError);
            BiasGradient = AccumulateGradients ? BiasGradient + outputError : outputError;
            
            return Weights[0] * outputError;
        }

        public override void DrainGradients()
        {
            WeightGradients[0] = Matrix<double>.Build.Dense(WeightGradients[0].RowCount, WeightGradients[0].ColumnCount);
            BiasGradient = Vector<double>.Build.Dense(BiasGradient.Count);
        }

        public override void SetGradientAccumulation(bool acc)
        {
            AccumulateGradients = acc;
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            switch (optimizerType)
            {
                case OptimizerType.GradientDescent:
                    Weights[0] -= learningRate * WeightGradients[0];
                    Bias -= learningRate * BiasGradient;
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate);
                    break;
            }

            if(AccumulateGradients)
                DrainGradients();
        }
    }
}
