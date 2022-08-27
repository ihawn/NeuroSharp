using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class SoftmaxActivationLayer : Layer
    {
        public SoftmaxActivationLayer(int? inputSize = null, int? outputSize = 0, int? id = null)
        {
            LayerType = LayerType.SoftmaxActivation;
            
            if (inputSize.HasValue)
                InputSize = inputSize.Value;
            if (outputSize.HasValue)
                OutputSize = outputSize.Value;
        }
        
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
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
