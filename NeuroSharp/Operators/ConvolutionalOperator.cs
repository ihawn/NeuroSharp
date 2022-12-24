using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using Newtonsoft.Json;
using System;


namespace NeuroSharp
{
    public class ConvolutionalOperator : ParameterizedLayer
    {
        [JsonIgnore]
        public ConvolutionalLayer ParentLayer { get; set; }
        
        private Adam _adam;
        
        [JsonProperty]
        private int _stride;
        [JsonProperty]
        private int _rawInputSize;
        [JsonProperty]
        private int _filters;
        [JsonProperty]
        private int _kernelSize;

        private int _filterOutputSize;
        private int _filterOutputDimension;
        private int _weightGradConvDimension;
        private int _inputGradConvDimension;
        private int _sliceSize;

        public ConvolutionalOperator(ConvolutionalLayer parent, int kernel, int filters, int stride = 1)
        {
            LayerType = LayerType.Convolutional;
            ParentLayer = parent;
            _kernelSize = kernel;
            _stride = stride;
            _filters = filters;
        }
        
        //json constructor
        public ConvolutionalOperator(Matrix<double>[] weights, int kernelSize,
            int stride, int inputSize, int outputSize, int filters)
        {
            Weights = weights;
            _kernelSize = kernelSize;
            _stride = stride;
            InputSize = inputSize;
            OutputSize = outputSize;
            _filters = filters;
            
            StoreConvolutionDimensionParameters();
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            int output = OutputSize;
            Input = input;
            Output = Vector<double>.Build.Dense(OutputSize);

            for (int i = 0; i < _filters; i++)
            {
                Output.SetSubVector(
                    i * _filterOutputSize, _filterOutputSize, 
                    Convolution(Input, Weights[i], _stride, _filterOutputDimension)
                );
            }
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Matrix<double> inputGradientMatrix = Matrix<double>.Build.Dense((int)Math.Sqrt(_rawInputSize), (int)Math.Sqrt(_rawInputSize));

            for(int i = 0; i < _filters; i++)
            {
                Vector<double> jacobianSlice = outputError.SubVector(i * _sliceSize, _sliceSize); // ∂L/∂Y

                WeightGradients[i] = AccumulateGradients ? 
                WeightGradients[i] +
                ComputeWeightGradient(Input, MathUtils.Unflatten(jacobianSlice), _stride, _weightGradConvDimension) :
                ComputeWeightGradient(Input, MathUtils.Unflatten(jacobianSlice), _stride, _weightGradConvDimension);

                Vector<double> singleGradient = 
                    ComputeInputGradient(Weights[i], MathUtils.Unflatten(jacobianSlice), _stride, _inputGradConvDimension);
                inputGradientMatrix += MathUtils.Unflatten(singleGradient);
            }

            return MathUtils.Flatten(inputGradientMatrix.Transpose()); //todo perform transpose implicitly in the convolution
        }

        public override void SetSizeIO()
        {
            _rawInputSize = ParentLayer.ChannelInputSize;
            InputSize = _rawInputSize;
            OutputSize = _filters * (int)Math.Pow((int)Math.Floor((Math.Sqrt(InputSize) - _kernelSize) / _stride + 1), 2);
            StoreConvolutionDimensionParameters();
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
                        Weights[i][x, y] = MathUtils.GetInitialWeightFromInputSize(InputSize);
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

        public void StoreConvolutionDimensionParameters()
        {
            _filterOutputDimension = MathUtils.GetConvolutionOutputSize(InputSize, _kernelSize, _stride);
            _filterOutputSize = (int)Math.Pow(_filterOutputDimension, 2);
            
            int weightsDim = (int)Math.Sqrt(OutputSize / _filters);
            int dilatedSize = weightsDim + (_stride - 1) * (weightsDim - 1);
            _weightGradConvDimension = MathUtils.GetConvolutionOutputSize(InputSize, dilatedSize, 1);
            
            
            int unpaddedSize = weightsDim + (_stride - 1) * (weightsDim - 1);
            int paddedSize = 2 * (_kernelSize - 1) + unpaddedSize;
            _inputGradConvDimension = MathUtils.GetConvolutionOutputSize(paddedSize * paddedSize, _kernelSize, 1);

            _sliceSize = OutputSize / _filters;
        }

        #region Operator Methods
        public static Vector<double> Convolution(Vector<double> flattenedImage, Matrix<double> weights, 
            int stride, int outDim)
        {
            int imageDim = (int)Math.Sqrt(flattenedImage.Count);
            Vector<double> output = Vector<double>.Build.Dense(outDim * outDim);
            
            for(int i = 0; i < outDim; i++)
            for (int j = 0; j < outDim; j++)
            for (int a = 0; a < weights.RowCount; a++)
            for (int b = 0; b < weights.RowCount; b++)
                output[i * outDim + j] += flattenedImage[j * stride + b + (i * stride + a) * imageDim] * weights[a, b];

            return output;
        }
        public static Vector<double> Convolution(Matrix<double> image, Matrix<double> weights, 
            int stride, int outDim)
        {
            Vector<double> output = Vector<double>.Build.Dense(outDim * outDim);

            int x;
            int y;
            for(int i = 0; i < outDim; i++)
            for (int j = 0; j < outDim; j++)
            for (int a = 0; a < weights.RowCount; a++)
            for (int b = 0; b < weights.RowCount; b++)
                output[i * outDim + j] += image[i * stride + a, j * stride + b] * weights[a, b];

            return output;
        }
        public static Matrix<double> MatrixTransposeConvolution(Vector<double> flattenedImage, Matrix<double> weights, 
            int stride, int outDim)
        {
            int imageDim = (int)Math.Sqrt(flattenedImage.Count);
            Matrix<double> output = Matrix<double>.Build.Dense(outDim, outDim);
            
            for(int i = 0; i < outDim; i++)
            for (int j = 0; j < outDim; j++)
            for (int a = 0; a < weights.RowCount; a++)
            for (int b = 0; b < weights.RowCount; b++)
                output[i, j] += flattenedImage[i * stride + b + (j * stride + a) * imageDim] * weights[a, b];

            return output;
        }
        
        // ∂L/∂W
        public static Matrix<double> ComputeWeightGradient(Vector<double> input, Matrix<double> outputJacobian, int stride, int outDim)
        {
            return MatrixTransposeConvolution(input, Dilate(outputJacobian, stride).Transpose(), stride: 1, outDim);
        }

        // ∂L/∂X
        public static Vector<double> ComputeInputGradient(Matrix<double> weight, Matrix<double> outputJacobian, int stride, int outDim)
        {
            return Convolution(
                    PadAndDilate(outputJacobian, stride, weight.RowCount), 
                    Rotate180(weight), stride: 1, outDim
                    );
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
