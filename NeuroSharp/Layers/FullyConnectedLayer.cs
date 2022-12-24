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
        private Adam _adam;

        public FullyConnectedLayer(int outputSize, int? inputSize = null)
        {
            LayerType = LayerType.FullyConnected;
            OutputSize = outputSize;

            if (inputSize != null)
                InputSize = inputSize.Value;
        }
        
        //json constructor
        public FullyConnectedLayer(Matrix<double> weight, Vector<double> bias,
            int inputSize, int outputSize, int id)
        {
            LayerType = LayerType.FullyConnected;
            Id = id;
            Weights = new Matrix<double>[] { weight };
            Biases = new Vector<double>[] { bias };
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Input * Weights[0] + Biases[0];
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            WeightGradients[0] = AccumulateGradients ? WeightGradients[0] + Input.OuterProduct(outputError) : Input.OuterProduct(outputError);
            BiasGradients[0] = AccumulateGradients ? BiasGradients[0] + outputError : outputError;
            return Weights[0] * outputError;
        }
        
        public override void InitializeParameters()
        {
            WeightGradients = new Matrix<double>[] { Matrix<double>.Build.Dense(InputSize, OutputSize) };
            Weights = new Matrix<double>[] { Matrix<double>.Build.Random(InputSize, OutputSize) };
            Biases = new Vector<double>[] { Vector<double>.Build.Random(OutputSize) };
            BiasGradients = new Vector<double>[] { Vector<double>.Build.Dense(OutputSize) };
            _adam = new Adam(InputSize, OutputSize);

            for (int i = 0; i < InputSize; i++)
                for (int j = 0; j < OutputSize; j++)
                    Weights[0][i, j] = MathUtils.GetInitialWeightFromInputSize(InputSize);
            for (int i = 0; i < OutputSize; i++)
                Biases[0][i] = MathUtils.GetInitialWeightFromInputSize(InputSize);
        }

        public override void DrainGradients()
        {
            WeightGradients[0] = Matrix<double>.Build.Dense(WeightGradients[0].RowCount, WeightGradients[0].ColumnCount);
            BiasGradients = new Vector<double>[] { Vector<double>.Build.Dense(BiasGradients[0].Count) };
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
                    Biases[0] -= learningRate * BiasGradients[0];
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
