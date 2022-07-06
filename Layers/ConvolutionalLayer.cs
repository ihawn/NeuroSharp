using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class ConvolutionalLayer : Layer
    {
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }

        //Adam containers
        public Matrix<float> MeanWeightGradient { get; set; }
        public Vector<float> MeanBiasGradient { get; set; }
        public Matrix<float> VarianceWeightGradient { get; set; }
        public Vector<float> VarianceBiasGradientt { get; set; }

        public ConvolutionalLayer(int inputSize, int outputSize, int convolutionSize)
        {
            Weights = Matrix<float>.Build.Random(convolutionSize, convolutionSize);
            Bias = Vector<float>.Build.Random(outputSize);

            for (int i = 0; i < convolutionSize; i++)
                for (int j = 0; j < convolutionSize; j++)
                    Weights[i, j] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input;
            Output = Convolution(Output, Weights).Item1;
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate)
        {
            //for now assume next layer has input which is a perfect square
            int dim = (int)Math.Round(Math.Sqrt(outputError.Count));
            Matrix<float> outGradientMatrix = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    outGradientMatrix[j, i] = outputError[i * dim + j];

            // ∂L/∂O -> Loss Gradient.   Equals gradient from previous layer, outGradientMatrix in this case
            // ∂L/∂F -> Weights Gradient. Equals convolution(Input, ∂L/∂O)
            // ∂L/∂X -> Input Gradient.  Equals full_convolution(WeightsTranspose, ∂L/∂O)

            var weightsGradient = Convolution(Input, outGradientMatrix, 0);
            Vector<float> inputGradient = BackwardsConvolution(Flatten(OrthoMatrix(Weights)), outGradientMatrix);

            AdamOutput a = Adam.Step(Weights, Bias, weightsGradient.Item2, outputError, sampleIndex + 1, MeanWeightGradient, MeanBiasGradient, VarianceWeightGradient, VarianceBiasGradientt, eta: learningRate);
            Weights = a.Weights;
            Bias = a.Bias;
            MeanWeightGradient = a.MeanWeightGradient;
            MeanBiasGradient = a.MeanBiasGradient;
            VarianceWeightGradient = a.VarianceWeightGradient;
            VarianceBiasGradientt = a.VarianceBiasGradient;

            return inputGradient;
        }

        // returns both flattened and unflattened gradient
        public static (Vector<float>, Matrix<float>) Convolution(Vector<float> flattenedImage, Matrix<float> Weights, /*float bias,*/ int stride = 1)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor((dim - (float)Weights.RowCount) / stride) + 1;

            Matrix<float> image = Matrix<float>.Build.Dense(dim, dim);
            for(int i = 0; i < dim; i++)
                for(int j = 0; j < dim; j++)
                    image[j, i] = flattenedImage[i*dim + j];

            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            int y = 0;
            int outY = 0;
            while (y + Weights.RowCount <= dim)
            {
                int x = 0;
                int outX = 0;
                while (x + Weights.ColumnCount <= dim)
                {
                    float sum = 0;
                    for (int n = 0; n < Weights.RowCount; n++)
                    {
                        for (int m = 0; m < Weights.RowCount; m++)
                        {
                            sum += Weights[m, n] * image[x + m, y + n]/* + bias*/;
                        }
                    }

                    output[outX, outY] = sum;
                    x += stride;
                    outX++;
                }

                y += stride;
                outY++;
            }


            return (Flatten(output), output);
        }

        public static Vector<float> BackwardsConvolution(Vector<float> flattenedImage, Matrix<float> Weights, int stride = 1)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int WeightsDim = Weights.RowCount;
            int outDim = dim + WeightsDim - 1;
            int paddedDim = dim + 2*(WeightsDim - 1);

            Matrix<float> image = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    image[j, i] = flattenedImage[i * dim + j];

            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            // build padding matrix. This will form the backwards convolution
            Matrix<float> padding = Matrix<float>.Build.Dense(paddedDim, paddedDim);
            int offset = outDim - dim;
            for (int i = 0; i < dim; i++)
            {
                for (int j = 0; j < dim; j++)
                {
                    padding[i + offset, j + offset] = image[i, j];
                }
            }

            // backward overlapping slide operation
            for (int i = outDim - 1; i >= 0; i--)
            {
                for(int j = outDim - 1; j >= 0; j--)
                {
                    float sum = 0;
                    for(int x = 0; x < WeightsDim; x++)
                    {
                        for(int y = 0; y < WeightsDim; y++)
                        {
                            sum += padding[y + j, x + i] * Weights[x, y];
                        }
                    }
                    output[outDim - i - 1, outDim - j - 1] = sum;
                }
            }

            output = output.Transpose();

            return Flatten(output);
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

        public static Vector<float> Flatten(Matrix<float> mtx)
        {
            List<float> f = new List<float>();
            for (int i = 0; i < mtx.RowCount; i++)
                for (int j = 0; j < mtx.ColumnCount; j++)
                    f.Add(mtx[i, j]);
            return Vector<float>.Build.DenseOfArray(f.ToArray());
        }

        public static Matrix<float> MaxPool(Vector<float> flattenedImage, int kernel = 2, int stride = 1)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor((dim - (float)kernel) / stride) + 1;

            Matrix<float> image = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    image[j, i] = flattenedImage[i * dim + j];

            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            int y = 0;
            int outY = 0;
            while (y + kernel <= dim)
            {
                int x = 0;
                int outX = 0;
                while (x + kernel <= dim)
                {
                    float max = float.MinValue;
                    for (int n = 0; n < kernel; n++)
                    {
                        for (int m = 0; m < kernel; m++)
                        {
                            if(image[x + m, y + n] > max)
                                max = image[x + m, y + n];
                        }
                    }

                    output[outX, outY] = max;
                    x += stride;
                    outX++;
                }

                y += stride;
                outY++;
            }

            return output;
        }
    }
}
