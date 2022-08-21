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
                            bias: JsonArrayToVector(layer["Bias"]),
                            weightGradient: JsonArrayToMatrix(layer["WeightGradients"][0]),
                            biasGradient: JsonArrayToVector(layer["BiasGradient"]),
                            adam: AdamFromJson(layer["_adam"]),
                            accumulateGradients: bool.Parse((string)layer["AccumulateGradients"])
                    ) : 
                GetLayerType(layer) == LayerType.Convolutional ? 
                    ConvolutionalLayerFromJson(layer) :
                GetLayerType(layer) == LayerType.MultiChannelConvolutional ?
                    new MultiChannelConvolutionalLayer(
                            operators: layer["ChannelOperators"]
                                .Select(conv => ConvolutionalLayerFromJson(conv)).ToArray(),
                            channelOutputs: JsonToArrayOfVectors(layer["_channelOutputs"]),
                            channelInputs: JsonToArrayOfVectors(layer["_channelInputs"]),
                            channelBackpropagationOutputs: JsonToArrayOfVectors(layer["_channelBackpropagationOutputs"]),
                            channelCount: Int32.Parse((string)layer["_channelCount"]),
                            channelInputSize: Int32.Parse((string)layer["_channelInputSize"]),
                            accumulateGradients: bool.Parse((string)layer["AccumulateGradients"])
                    ) :
                GetLayerType(layer) == LayerType.Activation ?
                    new ActivationLayer(
                        type: (ActivationType)Int32.Parse((string)layer["ActivationType"])
                    ) as Layer : 
                GetLayerType(layer) == LayerType.SoftmaxActivation ?
                    new SoftmaxActivationLayer() :
                GetLayerType(layer) == LayerType.MaxPooling ? 
                    new MaxPoolingLayer(
                        maxPoolPositions: layer["MaxPoolPositions"]
                            .Select(x =>
                                x.Select(y =>
                                    (Int32.Parse((string)y["Item1"]), Int32.Parse((string)y["Item2"]))
                                ).ToList()
                            ).ToList(),
                        poolSize: Int32.Parse((string)layer["_poolSize"]),
                        inputSize: Int32.Parse((string)layer["_inputSize"]),
                        outputSize: Int32.Parse((string)layer["_outputSize"]),
                        stride: Int32.Parse((string)layer["_stride"]),
                        filters: Int32.Parse((string)layer["_filters"])
                    ) : null
            ).ToList();
            
            string name = (string)jo["Name"];
            LossType lossType = (LossType)Int32.Parse((string)jo["LossType"]);
            
            return new Network(new List<Layer>(layers), lossType, name);
        }

        public override bool CanWrite
        {
            get { return false; }
        }

        public override void WriteJson(JsonWriter writer, object value, JsonSerializer serializer)
        {
            throw new NotImplementedException();
        }

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
                meanBiasGradient: JsonArrayToVector(obj["_meanBiasGradient"]),
                varianceWeightGradient: JsonToArrayOfMatrices(obj["_varianceWeightGradient"]),
                varianceBiasGradient: JsonArrayToVector(obj["_varianceBiasGradient"]),
                beta1: Double.Parse((string)obj["_beta1"]),
                beta2: Double.Parse((string)obj["_beta2"]),
                epsilon: Double.Parse((string)obj["_epsilon"])
            );
        }

        public LayerType GetLayerType(JToken obj)
        {
            return (LayerType)Int32.Parse((string)obj["LayerType"]);
        }

        public ConvolutionalLayer ConvolutionalLayerFromJson(JToken layer)
        {
            return new ConvolutionalLayer(
                weights: JsonToArrayOfMatrices(layer["Weights"]),
                weightGradients: JsonToArrayOfMatrices(layer["WeightGradients"]),
                kernelSize: Int32.Parse((string)layer["_kernelSize"]),
                stride: Int32.Parse((string)layer["_stride"]),
                inputSize: Int32.Parse((string)layer["_inputSize"]),
                outputSize: Int32.Parse((string)layer["_outputSize"]),
                filters: Int32.Parse((string)layer["_filters"]),
                adam: AdamFromJson(layer["_adam"]),
                accumulateGradients: bool.Parse((string)layer["AccumulateGradients"])
            );
        }
    }
}
