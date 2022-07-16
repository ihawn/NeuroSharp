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
            Weights = Matrix<double>.Build.Random(kernel, kernel);
            _adam = new Adam(kernel, kernel);
            _stride = stride;
            _inputSize = inputSize;
            _outputSize = (int)Math.Floor((inputSize - (double)kernel) / stride + 1);

            for (int i = 0; i < kernel; i++)
                for (int j = 0; j < kernel; j++)
                    Weights[i, j] = Utils.GetInitialWeight(inputSize);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Input;
            Output = Convolution(Output, Weights, _stride).Item1;
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError, OptimizerType optimzerType, int sampleIndex, double learningRate)
        {
            Matrix<double> outputJacobian = Utils.Unflatten(outputError); // ∂L/∂Y
            WeightGradient = ComputeWeightGradient(Input, outputJacobian, _stride);
            Vector<double> inputGradient = ComputeInputGradient(Weights, outputJacobian, _stride);

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


        #region Operator Methods
        public static (Vector<double>, Matrix<double>) Convolution(Vector<double> flattenedImage, Matrix<double> weights, int stride)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor(((double)dim - weights.RowCount) / stride) + 1;

            Matrix<double> image = Utils.Unflatten(flattenedImage);
            Matrix<double> output = Matrix<double>.Build.Dense(outDim, outDim);

            for(int i = 0; i < outDim; i++)
                for(int j = 0; j < outDim; j++)
                    for(int a = 0; a < weights.RowCount; a++)
                        for(int b = 0; b < weights.RowCount; b++)
                            output[i, j] += image[j * stride + b, i * stride + a] * weights[a, b];

            return (Utils.Flatten(output), output);
        }

        // ∂L/∂W
        public static Matrix<double> ComputeWeightGradient(Vector<double> input, Matrix<double> outputJacobian, int stride)
        {
            Matrix<double> dilatedGradient = Dilate(outputJacobian, stride);
            return Convolution(input, dilatedGradient, stride: 1).Item2;
        }

        // ∂L/∂X
        public static Vector<double> ComputeInputGradient(Matrix<double> weight, Matrix<double> outputJacobian, int stride)
        {
            Matrix<double> rotatedWeight = Rotate180(weight);
            Matrix<double> paddedDilatedGradient = PadAndDilate(outputJacobian, stride, rotatedWeight.RowCount);
            return Convolution(Utils.Flatten(paddedDilatedGradient), rotatedWeight, stride: 1).Item1;
        }


        public static Matrix<double> PadAndDilate(Matrix<double> passedGradient, int stride, int kernel)
        {
            int weightsDim = passedGradient.RowCount;
            int dilation = stride - 1;
            int padding = kernel - 1;
            int unpaddedSize = weightsDim + dilation * (weightsDim - 1);
            int outDim = 2 * padding + unpaddedSize;

            Matrix<double> paddedDilatedMatrix = Matrix<double>.Build.Dense(outDim, outDim);

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


        public static Matrix<double> Dilate(Matrix<double> passedGradient, int stride)
        {
            int dilation = stride - 1;
            int gradientDim = passedGradient.RowCount;
            int outDim = gradientDim + dilation * (gradientDim - 1);

            Matrix<double> dilatedMatrix = Matrix<double>.Build.Dense(outDim, outDim);

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


        public static Matrix<double> Rotate180(Matrix<double> mtx)
        {
            Matrix<double> temp = Matrix<double>.Build.Dense(mtx.RowCount, mtx.ColumnCount);
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    temp[i, j] = mtx[i, mtx.ColumnCount - j - 1];
            Matrix<double> output = Matrix<double>.Build.Dense(mtx.RowCount, mtx.ColumnCount);
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    output[i, j] = temp[mtx.RowCount - i - 1, j];
            return output;
        }
        #endregion
    }
}
