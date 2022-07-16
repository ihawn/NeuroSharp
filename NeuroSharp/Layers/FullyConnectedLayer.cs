﻿using MathNet.Numerics.LinearAlgebra;
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
            Weights = Matrix<double>.Build.Random(inputSize, outputSize);
            Bias = Vector<double>.Build.Random(outputSize);
            _adam = new Adam(inputSize, outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = Utils.GetInitialWeight(inputSize);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = Utils.GetInitialWeight(inputSize);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Input * Weights + Bias;
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError, OptimizerType optimzerType, int sampleIndex, double learningRate)
        {
            Vector<double> inputError = outputError * Weights.Transpose();
            WeightGradient = Input.OuterProduct(outputError);
            BiasGradient = outputError;

            switch (optimzerType)
            {
                case OptimizerType.GradientDescent:
                    Weights -= learningRate * WeightGradient;
                    Bias -= learningRate * outputError;
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate);
                    break;
            }

            return inputError;
        }
    }
}