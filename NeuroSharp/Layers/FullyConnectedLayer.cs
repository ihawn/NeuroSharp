using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;

namespace NeuroSharp
{
    public class FullyConnectedLayer : ParameterizedLayer
    {
        private Adam _adam;

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            WeightGradients = new Matrix<double>[1];
            Weights = new Matrix<double>[] { Matrix<double>.Build.Random(inputSize, outputSize) };
            Bias = Vector<double>.Build.Random(outputSize);
            _adam = new Adam(inputSize, outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[0][i, j] = Utils.GetInitialWeight(inputSize);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = Utils.GetInitialWeight(inputSize);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Input * Weights[0] + Bias;
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            WeightGradients[0] = Input.OuterProduct(outputError);
            BiasGradient = outputError;
            return Weights[0] * outputError;
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
        }
    }
}
