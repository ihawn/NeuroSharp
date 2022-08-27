﻿using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

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
            Vector<double> vec = Vector<double>.Build.Dense(Input.Count);
            for (int i = 0; i < Input.Count; i++)
                vec[i] = _activationPrime(Input[i]);
            return vec.PointwiseMultiply(outputError);
        }
    }
}
