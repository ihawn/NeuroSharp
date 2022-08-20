using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class MultiChannelConvolutionalLayer : ParameterizedLayer
    {
        public ConvolutionalLayer[] ChannelOperators { get; set; }
        [JsonProperty]
        private int _channelCount;
        [JsonProperty]
        private int _channelInputSize;

        [JsonProperty]
        private Vector<double>[] _channelOutputs;
        [JsonProperty]
        private Vector<double>[] _channelInputs;
        [JsonProperty]
        private Vector<double>[] _channelBackpropagationOutputs;

        public MultiChannelConvolutionalLayer(int inputSize, int kernel, int filters, int stride = 1, int channels = 1)
        {
            LayerType = LayerType.MultiChannelConvolutional;
            ChannelOperators = new ConvolutionalLayer[channels];
            _channelOutputs = new Vector<double>[channels];
            _channelInputs = new Vector<double>[channels];
            _channelBackpropagationOutputs = new Vector<double>[channels];
            _channelCount = channels;
            _channelInputSize = inputSize / _channelCount;

            for (int i = 0; i < channels; i++)
                ChannelOperators[i] = new ConvolutionalLayer(_channelInputSize, kernel, filters, stride);
        }
        
        //json constructor
        public MultiChannelConvolutionalLayer(ConvolutionalLayer[] operators, Vector<double>[] channelOutputs,
            Vector<double>[] channelInputs, Vector<double>[] channelBackpropagationOutputs, int channelCount,
            int channelInputSize, bool accumulateGradients)
        {
            ChannelOperators = operators;
            _channelOutputs = channelOutputs;
            _channelInputs = channelInputs;
            _channelBackpropagationOutputs = channelBackpropagationOutputs;
            _channelCount = channelCount;
            _channelInputSize = channelInputSize;
            SetGradientAccumulation(accumulateGradients);
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            _channelInputs = SplitInputToChannels(input, _channelCount, _channelInputSize);
            Input = input;

            Parallel.For(0, _channelCount, i =>
            {
                _channelOutputs[i] = ChannelOperators[i].ForwardPropagation(_channelInputs[i]);
            });

            Output = Vector<double>.Build.Dense(_channelOutputs[0].Count);
            for (int i = 0; i < _channelCount; i++)
                Output += _channelOutputs[i];
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Parallel.For(0, _channelCount, i =>
            {
                _channelBackpropagationOutputs[i] = ChannelOperators[i].BackPropagation(outputError);
            });

            return CombineChannelBackPropagation(_channelBackpropagationOutputs, _channelCount, _channelInputSize);
        }

        public override void DrainGradients()
        {
            for (int i = 0; i < _channelCount; i++)
                ChannelOperators[i].DrainGradients();
        }

        public override void SetGradientAccumulation(bool acc)
        {
            for (int i = 0; i < _channelCount; i++)
                ChannelOperators[i].SetGradientAccumulation(acc);
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            Parallel.For(0, _channelCount, i =>
            {
                ChannelOperators[i].UpdateParameters(optimizerType, sampleIndex, learningRate);
            });

            DrainGradients();
        }

        public static Vector<double>[] SplitInputToChannels(Vector<double> input, int channelCount, int channelInputSize)
        {
            Vector<double>[] channels = new Vector<double>[channelCount];
            for (int i = 0; i < channelCount; i++)
                channels[i] = input.SubVector(i * channelInputSize, channelInputSize);
            return channels;
        }

        public static Vector<double> CombineChannelBackPropagation(Vector<double>[] input, int channelCount, int channelInputSize)
        {
            Vector<double> output = Vector<double>.Build.Dense(channelCount * channelInputSize);
            for (int i = 0; i < channelCount; i++)
                for (int j = 0; j < channelInputSize; j++)
                    output[i * channelInputSize + j] = input[i][j];
            return output;
        }
    }
}
