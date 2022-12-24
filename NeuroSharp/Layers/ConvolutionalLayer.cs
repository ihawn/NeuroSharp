using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using System.Threading.Tasks;

namespace NeuroSharp
{
    public class ConvolutionalLayer : ParameterizedLayer
    {
        public ConvolutionalOperator[] ChannelOperators { get; set; }
        public int FilterCount { get; set; }
        public int ChannelInputSize { get; set; }
        public int ChannelCount { get; set; }
        
        private Vector<double>[] _channelOutputs;
        private Vector<double>[] _channelInputs;
        private Vector<double>[] _channelBackpropagationOutputs;

        public ConvolutionalLayer(int kernel, int filters, int stride = 1, int channels = 1)
        {
            LayerType = LayerType.Convolutional;
            ChannelOperators = new ConvolutionalOperator[channels];
            _channelOutputs = new Vector<double>[channels];
            _channelInputs = new Vector<double>[channels];
            _channelBackpropagationOutputs = new Vector<double>[channels];
            ChannelCount = channels;
            FilterCount = filters;

            for (int i = 0; i < channels; i++)
                ChannelOperators[i] = new ConvolutionalOperator(this, kernel, filters, stride);
        }
        
        //json constructor
        public ConvolutionalLayer(ConvolutionalOperator[] operators, int channelCount,
            int channelInputSize, int inputSize, int outputSize, int id)
        {
            ChannelOperators = operators;
            Id = id;
            _channelOutputs = new Vector<double>[channelCount];
            ChannelCount = channelCount;
            ChannelInputSize = channelInputSize;
            InputSize = inputSize;
            OutputSize = outputSize;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            _channelInputs = SplitInputToChannels(input, ChannelCount, ChannelInputSize);
            Input = input;

            Parallel.For(0, ChannelCount, i =>
            {
                _channelOutputs[i] = ChannelOperators[i].ForwardPropagation(_channelInputs[i]);
            });

            Output = Vector<double>.Build.Dense(_channelOutputs[0].Count);
            for (int i = 0; i < ChannelCount; i++)
                Output += _channelOutputs[i];
            return Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Parallel.For(0, ChannelCount, i =>
            {
                _channelBackpropagationOutputs[i] = ChannelOperators[i].BackPropagation(outputError);
            });

            return CombineChannelBackPropagation(_channelBackpropagationOutputs, ChannelCount, ChannelInputSize);
        }

        public override void SetSizeIO()
        {
            InputSize = Id > 0 ? ParentNetwork.Layers[Id - 1].OutputSize : ParentNetwork.EntrySize;
            ChannelInputSize = InputSize / ChannelCount;
            for (int i = 0; i < ChannelCount; i++)
                ChannelOperators[i].SetSizeIO();
            OutputSize = ChannelOperators[0].OutputSize;
        }
        
        public override void InitializeParameters()
        {
            for(int i = 0; i < ChannelCount; i++)
                ChannelOperators[i].InitializeParameters();
        }

        public override void DrainGradients()
        {
            for (int i = 0; i < ChannelCount; i++)
                ChannelOperators[i].DrainGradients();
        }

        public override void SetGradientAccumulation(bool acc)
        {
            for (int i = 0; i < ChannelCount; i++)
                ChannelOperators[i].SetGradientAccumulation(acc);
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            Parallel.For(0, ChannelCount, i =>
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
