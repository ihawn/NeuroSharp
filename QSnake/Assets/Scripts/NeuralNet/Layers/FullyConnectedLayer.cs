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
        public Matrix<float> WeightGradient { get; set; }
        public Vector<float> BiasGradient { get; set; }

        public FullyConnectedLayer(int inputSize, int outputSize)
        {
            #region hardcoded
            /*float[,] input1 = new float[,]
            {
                { 0.2227f, -0.0198f, -0.1030f,  0.0870f, -0.0629f, -0.0540f, -0.0798f,  0.1665f, -0.0068f,  0.0285f, -0.0188f },
                { 0.1218f,  0.1233f, -0.2650f, -0.0281f, -0.1503f,  0.1978f,  0.2136f,  0.0159f,  0.2188f, 0.2681f, -0.2405f },
                { 0.1548f, -0.2080f, -0.2429f,  0.2956f,  0.0951f, -0.1789f,  0.2437f,  0.0809f,  0.3009f,  0.0597f,  0.2489f },
                { -0.1085f, -0.2835f, -0.0254f, -0.0854f,  0.2432f, 0.0266f, -0.1151f, -0.2360f,  0.2140f, -0.0865f,  0.1179f }
            };
            float[] bias1 = new float[] { -0.1430f, -0.1882f, 0.0658f, -0.0196f };

            float[,] input2 = new float[,]
            {
                { -0.3025f, -0.2036f, -0.3880f,  0.2955f },
                { -0.1374f,  0.1184f, -0.4437f,  0.4445f },
                { 0.1989f,  0.2389f,  0.3519f, -0.1063f }
            };
            float[] bias2 = new float[] { -0.1978f, 0.0826f, 0.3677f };

            if(inputSize == 11)
            {
                Weights = Matrix<float>.Build.DenseOfArray(input1);
                Bias = Vector<float>.Build.DenseOfArray(bias1);
            }
            else
            {
                Weights = Matrix<float>.Build.DenseOfArray(input2);
                Bias = Vector<float>.Build.DenseOfArray(bias2);
            }*/
            #endregion

            Weights = Matrix<float>.Build.Random(inputSize, outputSize);
            Bias = Vector<float>.Build.Random(outputSize);

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = UnityEngine.Random.Range(-0.4f, 0.4f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = UnityEngine.Random.Range(-0.4f, 0.4f);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input * Weights + Bias;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError)
        {
            Vector<float> inputError = outputError * Weights.Transpose(); // ∂E/∂X
            WeightGradient = Input.OuterProduct(outputError);             // ∂E/∂W
            BiasGradient = outputError;                                   // ∂E/∂Y: note that this is the same as ∂E/∂B

            Adam a = new Adam(this);
            a.Step();

            return inputError;
        }
    }
}
