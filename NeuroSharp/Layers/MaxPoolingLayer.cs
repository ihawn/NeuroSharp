using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.MathUtils;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class MaxPoolingLayer : Layer
    {
        public List<List<(int, int)>> MaxPoolPositions { get; set; }

        private int _poolSize;
        private int _inputSize;
        private int _outputSize;
        private int _stride;
        private int _filters;

        public MaxPoolingLayer(int inputSize, int poolSize, int prevFilterCount, int stride = 1)
        {
            MaxPoolPositions = new List<List<(int, int)>>();
            _poolSize = poolSize;
            _inputSize = inputSize;
            int dim = (int)Math.Round(Math.Sqrt(inputSize/prevFilterCount));
            _outputSize = (int)Math.Floor(((double)dim - poolSize) / stride + 1);
            _stride = stride;
            _filters = prevFilterCount;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            MaxPoolPositions.Clear();

            List<Matrix<double>> featureMaps = SliceFlattenedMatrixIntoSquares(input, _filters);
            List<double> rawOutput = new List<double>();

            for(int i = 0; i < featureMaps.Count; i++)
            {
                var maxPoolResult = MaxPool(featureMaps[i], _poolSize, _stride);
                MaxPoolPositions.Add(maxPoolResult.Item2);
                foreach (double d in Utils.Flatten(maxPoolResult.Item1))
                    rawOutput.Add(d);
            }


            Output = Vector<double>.Build.DenseOfEnumerable(rawOutput);
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            int dim = (int)Math.Round(Math.Sqrt(_inputSize/_filters));
            int errorOffset = outputError.Count / _filters;
            List<double> rawBackpropData = new List<double>();

            for(int k = 0; k < _filters; k++)
            {
                Matrix<double> backwardsGradient = Matrix<double>.Build.Dense(dim, dim);
                for (int i = 0; i < MaxPoolPositions[k].Count; i++)
                {
                    var coord = new { x = MaxPoolPositions[k][i].Item1, y = MaxPoolPositions[k][i].Item2 };
                    backwardsGradient[coord.x, coord.y] += outputError[i + k * errorOffset];
                }
                foreach (double d in Utils.Flatten(backwardsGradient))
                    rawBackpropData.Add(d);
            }

            return Vector<double>.Build.DenseOfEnumerable(rawBackpropData);
        }

        public static (Matrix<double>, List<(int, int)>) MaxPool(Matrix<double> mtx, int poolSize, int stride)
        {
            int outDim = (int)Math.Floor(((double)mtx.RowCount - poolSize) / stride + 1);
            Matrix<double> output = Matrix<double>.Build.Dense(outDim, outDim);
            List<(int, int)> maxPositions = new List<(int, int)>();

            int outX = 0;
            for (int i = 0; i <= mtx.RowCount - poolSize; i += stride)
            {
                int outY = 0;
                for (int j = 0; j <= mtx.ColumnCount - poolSize; j += stride)
                {
                    double max = double.MinValue;
                    (int, int) argMax = (0, 0);
                    for (int y = j; y < j + poolSize; y++)
                    {
                        for (int x = i; x < i + poolSize; x++)
                        {
                            if (mtx[x, y] > max)
                            {
                                max = mtx[x, y];
                                argMax = (x, y);
                            }
                        }
                    }

                    maxPositions.Add(argMax);
                    output[outX, outY] = max;

                    outY++;
                }
                outX++;
            }

            return (output, maxPositions);
        }

        public static List<Matrix<double>> SliceFlattenedMatrixIntoSquares(Vector<double> vec, int slices)
        {
            int dim = (int)Math.Round(Math.Sqrt(vec.Count / slices));
            List<Matrix<double>> featureMaps = new List<Matrix<double>>();
            //Parallel.For(0, slices, i =>
            for (int i = 0; i < slices; i++)
            {
                Matrix<double> mtx = Matrix<double>.Build.Dense(dim, dim);
                for (int x = 0; x < dim; x++)
                {
                    for (int y = 0; y < dim; y++)
                    {
                        mtx[x, y] = vec[i * dim * dim + x * dim + y];
                    }
                }
                featureMaps.Add(mtx);
            }//);
            return featureMaps;
        }
    }
}