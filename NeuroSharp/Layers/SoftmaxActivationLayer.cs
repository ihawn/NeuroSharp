using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class SoftmaxActivationLayer : Layer
    {
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = ActivationFunctions.Softmax(input);
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError, OptimizerType optimzerType, int sampleIndex, double learningRate = 0.001f)
        {
            Matrix<double> jacobian = ActivationFunctions.SoftmaxPrime(Input);
            return outputError * jacobian;
        }
    }
}
