using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using System;

namespace NeuroSharp
{
    public class ActivationLayer : Layer
    {
        private Func<double, double> _activation;
        private Func<double, double> _activationPrime;

        public ActivationType ActivationType { get; set; }

        public ActivationLayer(ActivationType type, int? inputSize = null, int? outputSize = null, int? id = null)
        {
            LayerType = LayerType.Activation;
            ActivationType = type;
            switch(type)
            {
                case ActivationType.ReLu:
                    _activation = ActivationFunctions.Relu;
                    _activationPrime = ActivationFunctions.ReluPrime;
                    break;
                case ActivationType.Tanh:
                    _activation = ActivationFunctions.Tanh;
                    _activationPrime= ActivationFunctions.TanhPrime;
                    break;
                case ActivationType.Sigmoid:
                    _activation = ActivationFunctions.Sigmoid;
                    _activationPrime = ActivationFunctions.SigmoidPrime;
                    break;
                //todo: integrate softmax into ActivationLayer instead of having it be its own layer
            }

            if (inputSize.HasValue)
                InputSize = inputSize.Value;
            if (outputSize.HasValue)
                OutputSize = outputSize.Value;
            if (id.HasValue)
                Id = id.Value;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Vector<double>.Build.Dense(input.Count);
            for(int i = 0; i < input.Count; i++)
                Output[i] = _activation(input[i]);

            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            return GradientPass(Input).PointwiseMultiply(outputError);
        }

        public Vector<double> GradientPass(Vector<double> error)
        {
            Vector<double> vec = Vector<double>.Build.Dense(error.Count);
            for (int i = 0; i < error.Count; i++)
                vec[i] = _activationPrime(error[i]);
            return vec;
        }
    }
}
