using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Spatial.Euclidean;
using UnityEngine;

namespace NeuroSharp
{
    public class FullyConnectedLayer : Layer
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            Weights = Matrix<float>.Build.Random(inputSize, outputSize);
            Bias = Vector<float>.Build.Random(outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = UnityEngine.Random.Range(-0.2f, 0.2f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = UnityEngine.Random.Range(-0.2f, 0.2f);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input * Weights + Bias;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, float learningRate)
        {
            Vector<float> inputError = outputError * Weights.Transpose();
            Matrix<float> weightsError = Input.OuterProduct(outputError);

            Weights -= learningRate * weightsError;
            Bias -= learningRate * outputError;
            return inputError;
        }
    }
}
