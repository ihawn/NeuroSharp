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

        public ConvolutionalLayer(int inputSize, int outputSize, int convolutionSize, int stride = 1)
        {
            Weights = Matrix<float>.Build.Random(convolutionSize, convolutionSize);
            _adam = new Adam(convolutionSize, convolutionSize);
            _stride = stride;

            for (int i = 0; i < convolutionSize; i++)
                for (int j = 0; j < convolutionSize; j++)
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
            Matrix<float> outGradientMatrix = Utils.Unflatten(outputError);

            var weightsGradient = Convolution(Input, outGradientMatrix, _stride);
            Vector<float> inputGradient = BackwardsConvolution(Utils.Flatten(OrthoMatrix(Weights)), outGradientMatrix, _stride);
            WeightGradient = weightsGradient.Item2;

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

        // returns both flattened and unflattened convolution
        public static (Vector<float>, Matrix<float>) Convolution(Vector<float> flattenedImage, Matrix<float> weights, int stride)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor((dim - (float)weights.RowCount) / stride) + 1;

            Matrix<float> image = Utils.Unflatten(flattenedImage);
            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            for(int i = 0; i < outDim; i++)
                for(int j = 0; j < outDim; j++)
                    for(int a = 0; a < weights.RowCount; a++)
                        for(int b = 0; b < weights.RowCount; b++)
                            output[i, j] += image[j * stride + b, i * stride + a] * weights[a, b];

            return (Utils.Flatten(output), output);
        }

        public static Vector<float> BackwardsConvolution(Vector<float> flattenedImage, Matrix<float> weights, int stride)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int WeightsDim = weights.RowCount;
            int outDim = dim + WeightsDim - 1;
            int paddedDim = dim + 2*(WeightsDim - 1);

            Matrix<float> image = Utils.Unflatten(flattenedImage);
            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            // build padding matrix. This will form the backwards convolution
            Matrix<float> padding = Matrix<float>.Build.Dense(paddedDim, paddedDim);
            int offset = outDim - dim;
            for (int i = 0; i < dim; i++)
            //Parallel.For(0, dim, i =>
            {
                for (int j = 0; j < dim; j++)
                {
                    padding[i + offset, j + offset] = image[i, j];
                }
            }//);

            // backward overlapping slide operation
            //for (int i = outDim - 1; i >= 0; i -= stride)
            Parallel.For(0, outDim, i =>
            {
                for (int j = outDim - 1; j >= 0; j -= stride)
                {
                    float sum = 0;
                    for (int x = 0; x < WeightsDim; x++)
                    {
                        for (int y = 0; y < WeightsDim; y++)
                        {
                            sum += padding[y + j, x + i] * weights[x, y];
                        }
                    }
                    output[outDim - i - 1, outDim - j - 1] = sum;
                }
            });

            output = output.Transpose();

            return Utils.Flatten(output);
        }

        // "rotates" matrix 180 degrees
        public static Matrix<float> OrthoMatrix(Matrix<float> mtx)
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
