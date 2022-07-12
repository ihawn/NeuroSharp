using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class SoftmaxActivationLayer : Layer
    {
        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = ActivationFunctions.Softmax(input);
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate = 0.001f)
        {
            Matrix<float> jacobian = ActivationFunctions.SoftmaxPrime(Input);
            return jacobian * outputError;
        }
    }
}
