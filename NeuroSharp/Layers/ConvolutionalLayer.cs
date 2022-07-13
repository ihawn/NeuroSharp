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

            // ∂L/∂Y -> Loss Gradient.   Equals gradient from previous layer, outGradientMatrix in this case
            // ∂L/∂W -> Weights Gradient. Equals convolution(Input, ∂L/∂O)
            var weightsGradient = Convolution(Input, outGradientMatrix, _stride);

            // ∂L/∂X -> Input Gradient.  Equals full_convolution(WeightsTranspose, ∂L/∂O)
            Vector<float> inputGradient = BackwardsConvolution(OrthoMatrix(Weights), outGradientMatrix, _stride);
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

        public static Vector<float> BackwardsConvolution(Matrix<float> rotatedWeight, Matrix<float> gradient, int stride)
        {
            Matrix<float> paddedDilatedGradient = PadAndDilate(gradient, stride, rotatedWeight.RowCount);
            return Convolution(Utils.Flatten(paddedDilatedGradient), rotatedWeight, stride: 1).Item1;
        }

        public static Matrix<float> PadAndDilate(Matrix<float> passedGradient, int stride, int kernel)
        {
            int weightsDim = passedGradient.RowCount;
            int dilation = stride - 1;
            int padding = kernel - 1;
            int unpaddedSize = weightsDim + dilation * (weightsDim - 1);
            int outDim = 2 * padding + unpaddedSize;

            Matrix<float> paddedMtx = Matrix<float>.Build.Dense(outDim, outDim);

            int x = 0;
            for (int i = padding; i < outDim - padding; i += stride)
            {
                int y = 0;
                for (int j = padding; j < outDim - padding; j += stride)
                {
                    paddedMtx[i, j] = passedGradient[x, y];
                    y++;
                }
                x++;
            }

            return paddedMtx;
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
