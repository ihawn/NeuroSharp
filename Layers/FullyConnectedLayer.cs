using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;

namespace NeuroSharp
{
    public class FullyConnectedLayer : Layer
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }
        public Matrix<float> WeightGradient { get; set; }
        public Vector<float> BiasGradient { get; set; }

        //Adam containers
        public Matrix<float> MeanWeightGradient { get; set; }
        public Vector<float> MeanBiasGradient { get; set; }
        public Matrix<float> VarianceWeightGradient { get; set; }
        public Vector<float> VarianceBiasGradientt { get; set; }
        public int BatchCount { get; set; }

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            Weights = Matrix<float>.Build.Random(inputSize, outputSize);
            Bias = Vector<float>.Build.Random(outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);

            MeanWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            MeanBiasGradient = Vector<float>.Build.Dense(outputSize);
            VarianceWeightGradient = Matrix<float>.Build.Dense(inputSize, outputSize);
            VarianceBiasGradientt = Vector<float>.Build.Dense(outputSize);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input * Weights + Bias;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, float learningRate = 0.001f, int sampleIndex = 1)
        {
            Vector<float> inputError = outputError * Weights.Transpose();
            Matrix<float> weightsError = Input.OuterProduct(outputError);

            AdamOutput a = Adam.Step(Weights, Bias, weightsError, outputError, sampleIndex + 1, MeanWeightGradient, MeanBiasGradient, VarianceWeightGradient, VarianceBiasGradientt, eta: learningRate);
            Weights = a.Weights;
            Bias = a.Bias;
            MeanWeightGradient = a.MeanWeightGradient;
            MeanBiasGradient = a.MeanBiasGradient;
            VarianceWeightGradient = a.VarianceWeightGradient;
            VarianceBiasGradientt = a.VarianceBiasGradient;

            //gradient descent without minibatching
            //Weights -= learningRate * weightsError;
            //Bias -= learningRate * outputError;

            return inputError;
        }
    }
}
