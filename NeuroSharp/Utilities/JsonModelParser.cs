using System.Text.Json.Nodes;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using NeuroSharp.Optimizers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace NeuroSharp.Utilities
{
    public class JsonModelParser : JsonConverter
    {
        public override bool CanConvert(Type objectType)
        {
            return (objectType == typeof(Network));
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, 
            JsonSerializer serializer)
        {
            JObject jo = JObject.Load(reader);

            List<Layer> layers = jo["Layers"].Select(layer => 
                GetLayerType(layer) == LayerType.FullyConnected ? 
                    new FullyConnectedLayer(
                        weight: JsonArrayToMatrix(layer["Weights"][0]),
                        bias: JsonArrayToVector(layer["Biases"][0]),
                        weightGradient: JsonArrayToMatrix(layer["WeightGradients"][0]),
                        biasGradient: JsonArrayToVector(layer["BiasGradients"][0]),
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        adam: AdamFromJson(layer["_adam"]),
                        accumulateGradients: bool.Parse((string)layer["AccumulateGradients"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) : 
                GetLayerType(layer) == LayerType.Convolutional ?
                    new ConvolutionalLayer(
                        operators: layer["ChannelOperators"]
                            .Select(conv => ConvolutionalOperatorFromJson(conv)).ToArray(),
                        channelOutputs: JsonToArrayOfVectors(layer["_channelOutputs"]),
                        channelInputs: JsonToArrayOfVectors(layer["_channelInputs"]),
                        channelBackpropagationOutputs: JsonToArrayOfVectors(layer["_channelBackpropagationOutputs"]),
                        channelCount: Int32.Parse((string)layer["ChannelCount"]),
                        channelInputSize: Int32.Parse((string)layer["ChannelInputSize"]),
                        accumulateGradients: bool.Parse((string)layer["AccumulateGradients"]),
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) :
                GetLayerType(layer) == LayerType.Recurrent ?
                    new RecurrentLayer(
                        sequenceLength: Int32.Parse((string)layer["_sequenceLength"]),
                        vocabSize: Int32.Parse((string)layer["_vocabSize"]),
                        hiddenSize: Int32.Parse((string)layer["_hiddenSize"]),
                        stateActivationType: (ActivationType)Int32.Parse((string)layer["_stateActivationType"]),
                        stateInput: JsonToArrayOfVectors(layer["StateInput"]),
                        states: JsonToArrayOfVectors(layer["States"]),
                        outputs: JsonArrayToVector(layer["Outputs"]),
                        recurrentGradient: JsonArrayToVector(layer["RecurrentGradient"]),
                        weights: JsonToArrayOfMatrices(layer["Weights"]),
                        biases: JsonToArrayOfVectors(layer["Biases"]),
                        adam: AdamFromJson(layer["_adam"]),
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        accumulateGradients: bool.Parse((string)layer["AccumulateGradients"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) :
                GetLayerType(layer) == LayerType.Activation ?
                    new ActivationLayer(
                        type: (ActivationType)Int32.Parse((string)layer["ActivationType"]),
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) as Layer : 
                GetLayerType(layer) == LayerType.SoftmaxActivation ?
                    new SoftmaxActivationLayer(
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) :
                GetLayerType(layer) == LayerType.MaxPooling ? 
                    new MaxPoolingLayer(
                        maxPoolPositions: layer["MaxPoolPositions"]
                            .Select(x =>
                                x.Select(y =>
                                    (Int32.Parse((string)y["Item1"]), Int32.Parse((string)y["Item2"]))
                                ).ToList()
                            ).ToList(),
                        inputSize: Int32.Parse((string)layer["InputSize"]),
                        outputSize: Int32.Parse((string)layer["OutputSize"]),
                        poolSize: Int32.Parse((string)layer["_poolSize"]),
                        stride: Int32.Parse((string)layer["_stride"]),
                        filters: Int32.Parse((string)layer["_filters"]),
                        id: Int32.Parse((string)layer["Id"])
                    ) : null
            ).ToList(); //todo add recurrent layer support. Also check that binary crossentropy is supported

            //todo: support model saving without the intention to train further (will reduce model size)
            
            foreach (Layer layer in layers)
            {
                if (layer is ConvolutionalLayer)
                {
                    ConvolutionalLayer conv = (ConvolutionalLayer)layer;
                    foreach (ConvolutionalOperator convolutionalOperator in conv.ChannelOperators)
                    {
                        convolutionalOperator.ParentLayer = conv;
                    }
                }
            }
            
            string name = (string)jo["Name"];
            LossType lossType = (LossType)Int32.Parse((string)jo["LossType"]);
            int entrySize = Int32.Parse((string)jo["EntrySize"]);
            
            return new Network(layers, lossType, name, entrySize);
        }

        public override bool CanWrite
        {
            get { return false; }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer) { }

        public Vector<double> JsonArrayToVector(JToken obj)
        {
            return Vector<double>.Build.DenseOfEnumerable(obj.Select(x => Double.Parse((string)x)));
        }
        
        public Matrix<double> JsonArrayToMatrix(JToken obj)
        {
            return MathUtils.Unflatten(JsonArrayToVector(obj["Values"]), 
                rowCount: Int32.Parse((string)obj["RowCount"]),
                colCount: Int32.Parse((string)obj["ColumnCount"]));
        }

        public Vector<double>[] JsonToArrayOfVectors(JToken obj)
        {
            return obj.Select(x => JsonArrayToVector(x)).ToArray();
        }

        public Matrix<double>[] JsonToArrayOfMatrices(JToken obj)
        {
            return obj.Select(x => JsonArrayToMatrix(x)).ToArray();
        }

        public Adam AdamFromJson(JToken obj)
        {
            return new Adam(
                meanWeightGradient: JsonToArrayOfMatrices(obj["_meanWeightGradient"]),
                meanBiasGradient: JsonToArrayOfVectors(obj["_meanBiasGradient"]),
                varianceWeightGradient: JsonToArrayOfMatrices(obj["_varianceWeightGradient"]),
                varianceBiasGradient: JsonToArrayOfVectors(obj["_varianceBiasGradient"]),
                beta1: Double.Parse((string)obj["_beta1"]),
                beta2: Double.Parse((string)obj["_beta2"]),
                epsilon: Double.Parse((string)obj["_epsilon"])
            );
        }

        public LayerType GetLayerType(JToken obj)
        {
            return (LayerType)Int32.Parse((string)obj["LayerType"]);
        }

        public ConvolutionalOperator ConvolutionalOperatorFromJson(JToken layer)
        {
            return new ConvolutionalOperator(
                weights: JsonToArrayOfMatrices(layer["Weights"]),
                weightGradients: JsonToArrayOfMatrices(layer["WeightGradients"]),
                kernelSize: Int32.Parse((string)layer["_kernelSize"]),
                stride: Int32.Parse((string)layer["_stride"]),
                inputSize: Int32.Parse((string)layer["InputSize"]),
                outputSize: Int32.Parse((string)layer["OutputSize"]),
                filters: Int32.Parse((string)layer["_filters"]),
                adam: AdamFromJson(layer["_adam"]),
                accumulateGradients: bool.Parse((string)layer["AccumulateGradients"])
            );
        }
    }
}
