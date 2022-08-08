using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class SoftmaxActivationLayer : Layer
    {
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            LayerType = LayerType.SoftmaxActivation;
            Input = input;
            Output = ActivationFunctions.Softmax(input);
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Matrix<double> jacobian = ActivationFunctions.SoftmaxPrime(Input);
            return outputError * jacobian;
        }
    }
}
