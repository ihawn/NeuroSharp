using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using NeuroSharp.Optimizers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Linq;


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

            List<Layer> layers = new List<Layer>();
            foreach (var layer in jo["Layers"])
            {
                LayerType layerType = GetLayerType(layer);

                switch (layerType)
                {
                    case LayerType.FullyConnected:
                        layers.Add(FullyConnectedLayerFromJSON(layer));
                        break;
                    
                    case LayerType.Convolutional:
                        layers.Add(
                            new ConvolutionalLayer(
                                operators: layer["ChannelOperators"]
                                    .Select(conv => ConvolutionalOperatorFromJson(conv)).ToArray(),
                                channelCount: Int32.Parse((string)layer["ChannelCount"]),
                                channelInputSize: Int32.Parse((string)layer["ChannelInputSize"]),
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                id: Int32.Parse((string)layer["Id"])
                            )
                        );
                        break;
                    
                    case LayerType.Recurrent:
                        layers.Add(
                            new RecurrentLayer(
                                sequenceLength: Int32.Parse((string)layer["_sequenceLength"]),
                                vocabSize: Int32.Parse((string)layer["_vocabSize"]),
                                hiddenSize: Int32.Parse((string)layer["_hiddenSize"]),
                                stateActivationType: (ActivationType)Int32.Parse((string)layer["_stateActivationType"]),
                                stateInput: JsonToArrayOfVectors(layer["StateInput"]),
                                states: JsonToArrayOfVectors(layer["States"]),
                                outputs: JsonArrayToVector(layer["Outputs"]),
                                weights: JsonToArrayOfMatrices(layer["Weights"]),
                                biases: JsonToArrayOfVectors(layer["Biases"]),
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                id: Int32.Parse((string)layer["Id"])
                            )
                        );
                        break;
                    
                    case LayerType.Activation:
                        layers.Add(
                            new ActivationLayer(
                                type: (ActivationType)Int32.Parse((string)layer["ActivationType"]),
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                id: Int32.Parse((string)layer["Id"])
                            ) 
                        );
                        break;
                    
                    case LayerType.SoftmaxActivation:
                        layers.Add(
                            new SoftmaxActivationLayer(
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                id: Int32.Parse((string)layer["Id"])
                            ) 
                        );
                        break;
                    
                    case LayerType.MaxPooling:
                        layers.Add(
                            new MaxPoolingLayer(
                                maxPoolPositions: layer["MaxPoolPositions"]
                                    .Select(x =>
                                        x.Select(y =>
                                            new XYPair(Int32.Parse((string)y["x"]), Int32.Parse((string)y["y"]))
                                        ).ToList()
                                    ).ToList(),
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                poolSize: Int32.Parse((string)layer["_poolSize"]),
                                stride: Int32.Parse((string)layer["_stride"]),
                                filters: Int32.Parse((string)layer["_filters"]),
                                id: Int32.Parse((string)layer["Id"])
                            )
                        );
                        break;
                    
                    case LayerType.LSTM:
                        layers.Add(
                            new LSTMLayer(
                                hiddenUnits: Int32.Parse((string)layer["_hiddenUnits"]),
                                vocabSize: Int32.Parse((string)layer["_vocabSize"]),
                                sequenceLength: Int32.Parse((string)layer["_sequenceLength"]),
                                inputSize: Int32.Parse((string)layer["InputSize"]),
                                outputSize: Int32.Parse((string)layer["OutputSize"]),
                                lstmGates: new []
                                {
                                    FullyConnectedLayerFromJSON(layer["LSTMGates"][0]),
                                    FullyConnectedLayerFromJSON(layer["LSTMGates"][1]),
                                    FullyConnectedLayerFromJSON(layer["LSTMGates"][2]),
                                    FullyConnectedLayerFromJSON(layer["LSTMGates"][3]),
                                    FullyConnectedLayerFromJSON(layer["LSTMGates"][4]),
                                }
                            )
                        );
                        break;
                }
            }

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
            var rawData = jo["Data"];
            List<string> data = new List<string>();
            if (rawData != null)
                data = rawData.Select(x => (string)x).ToList();
            
            return new Network(layers, lossType, name, entrySize, data);
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
        
        public FullyConnectedLayer FullyConnectedLayerFromJSON(JToken layer)
        {
            return new FullyConnectedLayer(
                weight: JsonArrayToMatrix(layer["Weights"][0]),
                bias: JsonArrayToVector(layer["Biases"][0]),
                inputSize: Int32.Parse((string)layer["InputSize"]),
                outputSize: Int32.Parse((string)layer["OutputSize"]),
                id: Int32.Parse((string)layer["Id"])
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
                kernelSize: Int32.Parse((string)layer["_kernelSize"]),
                stride: Int32.Parse((string)layer["_stride"]),
                inputSize: Int32.Parse((string)layer["InputSize"]),
                outputSize: Int32.Parse((string)layer["OutputSize"]),
                filters: Int32.Parse((string)layer["_filters"])
            );
        }
    }
}
