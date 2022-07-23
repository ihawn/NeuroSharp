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
        private int _filters;

        public ConvolutionalLayer(int inputSize, int kernel, int filters, int stride = 1)
        {
            WeightGradient = new Matrix<double>[filters];
            Weights = new Matrix<double>[filters];

            for (int i = 0; i < filters; i++)
            {
                Weights[i] = Matrix<double>.Build.Random(kernel, kernel);
                for (int x = 0; x < kernel; x++)
                    for (int y = 0; y < kernel; y++)
                        Weights[i][x, y] = Utils.GetInitialWeight(inputSize);
            }

            _adam = new Adam(kernel, kernel, weightCount: filters);
            _stride = stride;
            _inputSize = filters * inputSize;
            _outputSize = filters * (int)Math.Pow((int)Math.Floor((Math.Sqrt(inputSize) - (double)kernel) / stride + 1), 2);
            _filters = filters;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Output = Vector<double>.Build.Dense(_outputSize);
            //Parallel.For(0, _filters, i =>
            for (int i = 0; i < _filters; i++)
            {
                Vector<double> singleFilterOutput = Convolution(Input, Weights[i], _stride).Item1;
                for (int j = 0; j < singleFilterOutput.Count; j++)
                    Output[i * singleFilterOutput.Count + j] = singleFilterOutput[j];
            }//);
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError, OptimizerType optimzerType, int sampleIndex, double learningRate)
        {
            Vector<double> inputGradient = Vector<double>.Build.Dense(_inputSize);
            Vector<double>[] jacobianSlices = new Vector<double>[_filters];

            //Parallel.For(0, _filters, i =>
            for (int i = 0; i < _filters; i++)
            {
                jacobianSlices[i] = Vector<double>.Build.Dense(outputError.Count / _filters); // ∂L/∂Y
                for (int j = 0; j < jacobianSlices[i].Count; j++)
                    jacobianSlices[i][j] = outputError[i * jacobianSlices[i].Count + j];
                WeightGradient[i] = ComputeWeightGradient(Input, Utils.Unflatten(jacobianSlices[i]), _stride);

                Vector<double> singleGradient = ComputeInputGradient(Weights[i], Utils.Unflatten(jacobianSlices[i]), _stride);
                for (int j = 0; j < singleGradient.Count; j++)
                    inputGradient[i * singleGradient.Count + j] = singleGradient[j];
            }//);

            switch (optimzerType)
            {
                case OptimizerType.GradientDescent:
                    for(int i = 0; i < _filters; i++)
                        Weights[i] -= learningRate * WeightGradient[i];
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

            //Parallel.For(0, outDim, i =>
            for(int i = 0; i < outDim; i++)
            {
                for (int j = 0; j < outDim; j++)
                    for (int a = 0; a < weights.RowCount; a++)
                        for (int b = 0; b < weights.RowCount; b++)
                            output[i, j] += image[j * stride + b, i * stride + a] * weights[a, b];
            }//);


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

            return paddedDilatedMatrix.Transpose();
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
