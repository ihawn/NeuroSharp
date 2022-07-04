using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class ConvolutionalLayer : Layer
    {
        public Matrix<float> Filter { get; set; }
        public Matrix<float> Weights { get; set; }
        public Vector<float> Bias { get; set; }

        private int _dimension;

        public ConvolutionalLayer(int inputSize, int outputSize, int convolutionSize)
        {
            Weights = Matrix<float>.Build.Random(inputSize, outputSize);
            Bias = Vector<float>.Build.Random(outputSize);
            _dimension = (int)Math.Round(Math.Sqrt(inputSize));

            for (int i = 0; i < inputSize; i++)
                for (int j = 0; j < outputSize; j++)
                    Weights[i, j] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
            for (int i = 0; i < outputSize; i++)
                Bias[i] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);

            Filter = Matrix<float>.Build.Random(convolutionSize, convolutionSize);
            for (int x = 0; x < convolutionSize; x++)
                for (int y = 0; y < convolutionSize; y++)
                    Filter[x, y] = MathUtils.Utils.NextFloat(-0.5f, 0.5f);
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;
            Output = Input;
            Output = Convolution(Output, Filter);
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
            // ∂L/∂F -> Filter Gradient. Equals convolution(Input, ∂L/∂O)
            // ∂L/∂X -> Input Gradient.  Equals full_convolution(FilterTranspose, ∂L/∂O)

            Vector<float> filterGradient = Convolution(Input, outGradientMatrix, 0);
            Vector<float> inputGradient = FullConvolution(Flatten(OrthoMatrix(Filter)), outGradientMatrix);

            // todo: do the update step here

            return inputGradient;
        }

        public static Vector<float> Convolution(Vector<float> flattenedImage, Matrix<float> filter, /*float bias,*/ int stride = 1)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = (int)Math.Floor((dim - (float)filter.RowCount) / stride) + 1;

            Matrix<float> image = Matrix<float>.Build.Dense(dim, dim);
            for(int i = 0; i < dim; i++)
                for(int j = 0; j < dim; j++)
                    image[j, i] = flattenedImage[i*dim + j];

            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);

            int y = 0;
            int outY = 0;
            while (y + filter.RowCount <= dim)
            {
                int x = 0;
                int outX = 0;
                while (x + filter.ColumnCount <= dim)
                {
                    float sum = 0;
                    for (int n = 0; n < filter.RowCount; n++)
                    {
                        for (int m = 0; m < filter.RowCount; m++)
                        {
                            sum += filter[m, n] * image[x + m, y + n]/* + bias*/;
                        }
                    }

                    output[outX, outY] = sum;
                    x += stride;
                    outX++;
                }

                y += stride;
                outY++;
            }

            List<float> rawOutput = new List<float>();
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    rawOutput.Add(output[i, j]);

            return Vector<float>.Build.DenseOfArray(rawOutput.ToArray());
        }

        public static Vector<float> FullConvolution(Vector<float> flattenedImage, Matrix<float> filter, int stride = 1)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            int outDim = dim + 2*(filter.RowCount - 1);

            Matrix<float> image = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    image[j, i] = flattenedImage[i * dim + j];

            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);


            for(int i = 0; i < outDim; i++)
            {
                for(int j = 0; j < outDim; j++)
                {
                    // todo: left to right/bottom to top sliding convolution
                }
            }


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
