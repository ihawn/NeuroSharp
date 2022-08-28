using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class FullyConnectedLayer : ParameterizedLayer
    {
        [JsonProperty]
        private Adam _adam;

        public FullyConnectedLayer(int outputSize)
        {
            LayerType = LayerType.FullyConnected;
            OutputSize = outputSize;
        }
        
        //json constructor
        public FullyConnectedLayer(Matrix<double> weight, Vector<double> bias, Matrix<double> weightGradient,
            Vector<double> biasGradient, int inputSize, int outputSize, Adam adam, bool accumulateGradients, int id)
        {
            LayerType = LayerType.FullyConnected;
            Id = id;
            Weights = new Matrix<double>[] { weight };
            WeightGradients = new Matrix<double>[] { weightGradient };
            Bias = bias;
            BiasGradient = biasGradient;
            InputSize = inputSize;
            OutputSize = outputSize;
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
        
        public override void InitializeParameters()
        {
            WeightGradients = new Matrix<double>[] { Matrix<double>.Build.Dense(InputSize, OutputSize) };
            Weights = new Matrix<double>[] { Matrix<double>.Build.Random(InputSize, OutputSize) };
            Bias = Vector<double>.Build.Random(OutputSize);
            BiasGradient = Vector<double>.Build.Dense(OutputSize);
            _adam = new Adam(InputSize, OutputSize);

            for (int i = 0; i < InputSize; i++)
                for (int j = 0; j < OutputSize; j++)
                    Weights[0][i, j] = MathUtils.GetInitialWeightFromInputSize(InputSize);
            for (int i = 0; i < OutputSize; i++)
                Bias[i] = MathUtils.GetInitialWeightFromInputSize(InputSize);
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
