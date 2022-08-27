using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Distributions;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class ConvolutionalOperator : ParameterizedLayer
    {
        public ConvolutionalLayer ParentLayer { get; set; }
        
        [JsonProperty]
        private Adam _adam;
        [JsonProperty]
        private int _stride;
        [JsonProperty]
        private int _rawInputSize;
        [JsonProperty]
        private int _filters;
        [JsonProperty]
        private int _kernelSize;

        public ConvolutionalOperator(ConvolutionalLayer parent, int kernel, int filters, int stride = 1)
        {
            LayerType = LayerType.Convolutional;
            ParentLayer = parent;
            _kernelSize = kernel;
            _stride = stride;
            _filters = filters;
        }
        
        //json constructor
        public ConvolutionalOperator(Matrix<double>[] weights, Matrix<double>[] weightGradients, int kernelSize,
            int stride, int inputSize, int outputSize, int filters, Adam adam, bool accumulateGradients)
        {
            LayerType = LayerType.Convolutional;
            Weights = weights;
            WeightGradients = weightGradients;
            _kernelSize = kernelSize;
            _stride = stride;
            InputSize = inputSize;
            OutputSize = outputSize;
            _filters = filters;
            _adam = adam;
            SetGradientAccumulation(accumulateGradients);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            int output = OutputSize;
            Input = input;
            Output = Vector<double>.Build.Dense(OutputSize);

            for (int i = 0; i < _filters; i++)
            {
                Vector<double> singleFilterOutput = Convolution(Input, Weights[i], _stride).Item1;
                for (int j = 0; j < singleFilterOutput.Count; j++)
                    Output[i * singleFilterOutput.Count + j] = singleFilterOutput[j];
            }
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Vector<double> inputGradient = Vector<double>.Build.Dense(InputSize);
            Vector<double>[] jacobianSlices = new Vector<double>[_filters];
            Matrix<double> inputGradientMatrix = Matrix<double>.Build.Dense((int)Math.Sqrt(_rawInputSize), (int)Math.Sqrt(_rawInputSize));
            Matrix<double>[] inputGradientPieces = new Matrix<double>[_filters];

            for(int i = 0; i < _filters; i++)
            {
                jacobianSlices[i] = Vector<double>.Build.Dense(outputError.Count / _filters); // ∂L/∂Y
                for (int j = 0; j < jacobianSlices[i].Count; j++)
                    jacobianSlices[i][j] = outputError[i * jacobianSlices[i].Count + j];

                WeightGradients[i] = AccumulateGradients ? 
                WeightGradients[i] + ComputeWeightGradient(Input, MathUtils.Unflatten(jacobianSlices[i]), _stride) :
                ComputeWeightGradient(Input, MathUtils.Unflatten(jacobianSlices[i]), _stride);

                Vector<double> singleGradient = ComputeInputGradient(Weights[i], MathUtils.Unflatten(jacobianSlices[i]), _stride);
                inputGradientPieces[i] = MathUtils.Unflatten(singleGradient);
            }

            for (int i = 0; i < _filters; i++)
                inputGradientMatrix += inputGradientPieces[i];

            return MathUtils.Flatten(inputGradientMatrix.Transpose());
        }

        public override void SetSizeIO()
        {
            _rawInputSize = ParentLayer.ChannelInputSize;
            InputSize = _rawInputSize;// * _filters;
            OutputSize = _filters * (int)Math.Pow((int)Math.Floor((Math.Sqrt(InputSize) - (double)_kernelSize) / _stride + 1), 2);
        }

        public override void InitializeParameters()
        {
            WeightGradients = new Matrix<double>[_filters];
            Weights = new Matrix<double>[_filters];

            for (int i = 0; i < _filters; i++)
            {
                Weights[i] = Matrix<double>.Build.Random(_kernelSize, _kernelSize);
                WeightGradients[i] = Matrix<double>.Build.Dense(_kernelSize, _kernelSize);
                for (int x = 0; x < _kernelSize; x++)
                    for (int y = 0; y < _kernelSize; y++)
                        Weights[i][x, y] = MathUtils.GetInitialWeight(InputSize);
            }
            
            _adam = new Adam(_kernelSize, _kernelSize, weightCount: _filters);
        }

        public override void DrainGradients()
        {
            for (int i = 0; i < _filters; i++)
                WeightGradients[i] = Matrix<double>.Build.Dense(_kernelSize, _kernelSize);
        }

        public override void SetGradientAccumulation(bool acc)
        {
            AccumulateGradients = acc;
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            switch (optimizerType)
            {
                case OptimizerType.GradientDescent:
                    for (int i = 0; i < _filters; i++)
                        Weights[i] -= learningRate * WeightGradients[i];
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate, includeBias: false);
                    break;
            }

            if(AccumulateGradients)
                DrainGradients();
        }


        #region Operator Methods
        public static (Vector<double>, Matrix<double>) Convolution(Vector<double> flattenedImage, Matrix<double> weights, int stride, bool transposeOutput = false)
        {
            int dim = (int)Math.Round(Math.Sqrt(flattenedImage.Count));
            double imageQuotient = ((double)dim - weights.RowCount) / stride + 1;
            int outDim = (int)Math.Floor(imageQuotient);

            Matrix<double> image = MathUtils.Unflatten(flattenedImage);

            Matrix<double> output = Matrix<double>.Build.Dense(outDim, outDim);

            //Parallel.For(0, outDim, i =>
            for(int i = 0; i < outDim; i++)
            {
                for (int j = 0; j < outDim; j++)
                    for (int a = 0; a < weights.RowCount; a++)
                        for (int b = 0; b < weights.RowCount; b++)
                            output[i, j] += image[j * stride + b, i * stride + a] * weights[a, b];
            }//);

            if(transposeOutput)
                output = output.Transpose();

            return (Utilities.MathUtils.Flatten(output), output);
        }

        // ∂L/∂W
        public static Matrix<double> ComputeWeightGradient(Vector<double> input, Matrix<double> outputJacobian, int stride)
        {
            Matrix<double> dilatedGradient = Dilate(outputJacobian, stride).Transpose();
            return Convolution(input, dilatedGradient, stride: 1, transposeOutput: true).Item2;
        }

        // ∂L/∂X
        public static Vector<double> ComputeInputGradient(Matrix<double> weight, Matrix<double> outputJacobian, int stride)
        {
            Matrix<double> rotatedWeight = Rotate180(weight);
            Matrix<double> paddedDilatedGradient = PadAndDilate(outputJacobian, stride, rotatedWeight.RowCount);
            return Convolution(MathUtils.Flatten(paddedDilatedGradient), rotatedWeight, stride: 1).Item1;
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
        
        //todo: add support for non-square images
    }
}
