using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;

namespace NeuroSharp
{
    public class ConvolutionalLayer : ParameterizedLayer
    {
        private Adam _adam;
        private int _stride;
        private int _outputSize;
        private int _inputSize;

        public ConvolutionalLayer(int inputSize, int kernel, int stride = 1)
        {
            Weights = Matrix<float>.Build.Random(kernel, kernel);
            _adam = new Adam(kernel, kernel);
            _stride = stride;
            _inputSize = inputSize;
            _outputSize = (int)Math.Floor((inputSize - (float)kernel) / stride + 1);

            for (int i = 0; i < kernel; i++)
                for (int j = 0; j < kernel; j++)
                    Weights[i, j] = Utils.GetInitialWeight(inputSize);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input;
            Output = Convolution(Output, Weights, _stride).Item1;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate)
        {
            Matrix<float> outputJacobian = Utils.Unflatten(outputError); // ∂L/∂Y
            WeightGradient = ComputeWeightGradient(Input, outputJacobian, _stride);
            Vector<float> inputGradient = ComputeInputGradient(Weights, outputJacobian, _stride);

            switch (optimzerType)
            {
                case OptimizerType.GradientDescent:
                    Weights -= learningRate * WeightGradient;
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate, includeBias: false);
                    break;
            }

            return inputGradient;
        }

        public static (Vector<float>, Matrix<float>) Convolution(Vector<float> flattenedImage, Matrix<float> weights, int stride)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor(((float)dim - weights.RowCount) / stride) + 1;

            Matrix<float> image = Utils.Unflatten(flattenedImage);
            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            for(int i = 0; i < outDim; i++)
                for(int j = 0; j < outDim; j++)
                    for(int a = 0; a < weights.RowCount; a++)
                        for(int b = 0; b < weights.RowCount; b++)
                            output[i, j] += image[j * stride + b, i * stride + a] * weights[a, b];

            return (Utils.Flatten(output), output);
        }

        // ∂L/∂W
        public static Matrix<float> ComputeWeightGradient(Vector<float> input, Matrix<float> outputJacobian, int stride)
        {
            Matrix<float> dilatedGradient = Dilate(outputJacobian, stride);
            return Convolution(input, dilatedGradient, stride: 1).Item2;
        }

        // ∂L/∂X
        public static Vector<float> ComputeInputGradient(Matrix<float> weight, Matrix<float> outputJacobian, int stride)
        {
            Matrix<float> rotatedWeight = Rotate180(weight);
            Matrix<float> paddedDilatedGradient = PadAndDilate(outputJacobian, stride, rotatedWeight.RowCount);
            return Convolution(Utils.Flatten(paddedDilatedGradient), rotatedWeight, stride: 1).Item1;
        }

        public static Matrix<float> PadAndDilate(Matrix<float> passedGradient, int stride, int kernel)
        {
            int weightsDim = passedGradient.RowCount;
            int dilation = stride - 1;
            int padding = kernel - 1;
            int unpaddedSize = weightsDim + dilation * (weightsDim - 1);
            int outDim = 2 * padding + unpaddedSize;

            Matrix<float> paddedDilatedMatrix = Matrix<float>.Build.Dense(outDim, outDim);

            int x = 0;
            for (int i = padding; i < outDim - padding; i += stride)
            {
                int y = 0;
                for (int j = padding; j < outDim - padding; j += stride)
                {
                    paddedDilatedMatrix[i, j] = passedGradient[x, y];
                    y++;
                }
                x++;
            }

            return paddedDilatedMatrix;
        }

        public static Matrix<float> Dilate(Matrix<float> passedGradient, int stride)
        {
            int dilation = stride - 1;
            int gradientDim = passedGradient.RowCount;
            int outDim = gradientDim + dilation * (gradientDim - 1);

            Matrix<float> dilatedMatrix = Matrix<float>.Build.Dense(outDim, outDim);

            int x = 0;
            for(int i = 0; i < outDim; i += stride)
            {
                int y = 0;
                for(int j = 0; j < outDim; j += stride)
                {
                    dilatedMatrix[i, j] = passedGradient[x, y];
                    y++;
                }
                x++;
            }

            return dilatedMatrix;
        }

        public static Matrix<float> Rotate180(Matrix<float> mtx)
        {
            Matrix<float> temp = Matrix<float>.Build.Dense(mtx.RowCount, mtx.ColumnCount);
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    temp[i, j] = mtx[i, mtx.ColumnCount - j - 1];
            Matrix<float> output = Matrix<float>.Build.Dense(mtx.RowCount, mtx.ColumnCount);
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    output[i, j] = temp[mtx.RowCount - i - 1, j];
            return output;
        }
    }
}
