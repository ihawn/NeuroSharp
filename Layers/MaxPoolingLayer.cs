using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.MathUtils;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class MaxPoolingLayer : Layer
    {
        public List<(int, int)> MaxPoolPositions { get; set; }

        private int _poolSize;
        private int _inputSize;
        private int _outputSize;
        private int _stride;

        public MaxPoolingLayer(int inputSize, int poolSize, int stride = 1)
        {
            MaxPoolPositions = new List<(int, int)>();
            _poolSize = poolSize;
            _inputSize = inputSize;
            int dim = (int)Math.Round(Math.Sqrt(inputSize));
            _outputSize = (int)Math.Floor(((float)dim - poolSize) / stride + 1);
            _stride = stride;
        }

        public override Vector<float> ForwardPropagation(Vector<float> input)
        {
            Input = input;

            int dim = (int)Math.Round(Math.Sqrt(input.Count));
            Matrix<float> inputMatrix = Matrix<float>.Build.Dense(dim, dim);
            for (int i = 0; i < dim; i++)
                for (int j = 0; j < dim; j++)
                    inputMatrix[j, i] = input[i * dim + j];

            Output = Utils.Flatten(MaxPool(inputMatrix, _poolSize, _stride));
            return Output;
        }

        public override Vector<float> BackPropagation(Vector<float> outputError, OptimizerType optimzerType, int sampleIndex, float learningRate)
        {
            int dim = (int)Math.Round(Math.Sqrt(_inputSize));
            Matrix<float> backwardsGradient = Matrix<float>.Build.Dense(dim, dim);
            for(int i = 0; i < MaxPoolPositions.Count; i++)
            {
                var coord = new { x = MaxPoolPositions[i].Item1, y = MaxPoolPositions[i].Item2 };
                backwardsGradient[coord.x, coord.y] = outputError[i];
            }

            return Utils.Flatten(backwardsGradient);
        }

        public Matrix<float> MaxPool(Matrix<float> mtx, int poolSize, int stride)
        {
            int outDim = (int)Math.Floor(((float)mtx.RowCount - poolSize) / stride + 1);
            Matrix<float> output = Matrix<float>.Build.Dense(outDim, outDim);
            MaxPoolPositions.Clear();

            int outX = 0;
            for(int i = 0; i <= mtx.RowCount - poolSize; i += stride)
            {
                int outY = 0;
                for (int j = 0; j <= mtx.ColumnCount - poolSize; j += stride)
                {
                    float max = float.MinValue;
                    (int, int) argMax = (0, 0);
                    for(int y = j; y < j + poolSize; y++)
                    {
                        for(int x = i; x < i + poolSize; x++)
                        {
                            if (mtx[x, y] > max)
                            {
                                max = mtx[x, y];
                                argMax = (x, y);
                            }
                        }
                    }

                    MaxPoolPositions.Add(argMax);
                    output[outX, outY] = max;

                    outY++;
                }
                outX++;
            }

            output = output.Transpose();
            return output;
        }
    }
}
