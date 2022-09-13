using System.Diagnostics;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp.Training
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public int EntrySize { get; set; }
        
        [JsonIgnore]
        public List<ParameterizedLayer> ParameterizedLayers { get; set; }
        
        [JsonIgnore]
        public Func<Vector<double>, Vector<double>, double> Loss { get; set; }
        [JsonIgnore]
        public Func<Vector<double>, Vector<double>, Vector<double>> LossPrime { get; set; }
        
        public LossType LossType { get; set; }
        public string Name { get; set; }

        public Network(int entrySize, string name = "")
        {
            Layers = new List<Layer>();
            Name = name;
            EntrySize = entrySize;
        }
        
        //json constructor
        public Network(List<Layer> layers, LossType lossType, string name, int entrySize)
        {
            Name = name;
            Layers = new List<Layer>();
            EntrySize = entrySize;
            foreach (Layer layer in layers)
            {
                layer.ParentNetwork = this;
                Layers.Add(layer);
            }

            UseLoss(lossType);
        }

        public void Add(Layer layer)
        {
            layer.Id = Layers.Count();
            layer.ParentNetwork = this;
            layer.SetSizeIO();
            if(layer is ParameterizedLayer)
                ((ParameterizedLayer)layer).InitializeParameters();
            Layers.Add(layer);
        }

        public void UseLoss(LossType loss)
        {
            LossType = loss;
            switch (loss)
            {
                case LossType.MeanSquaredError:
                    Loss = LossFunctions.MeanSquaredError;
                    LossPrime = LossFunctions.MeanSquaredErrorPrime;
                    break;
                case LossType.CategoricalCrossentropy:
                    Loss = LossFunctions.CategoricalCrossentropy;
                    LossPrime = LossFunctions.CategoricalCrossentropyPrime;
                    break;
                case LossType.BinaryCrossentropy:
                    Loss = LossFunctions.BinaryCrossentropy;
                    LossPrime = LossFunctions.CategoricalCrossentropyPrime;
                    break;
            }
        }

        public Vector<double> Predict(Vector<double> inputData)
        {
            Vector<double> output = inputData;
            foreach (var layer in Layers)
                output = layer.ForwardPropagation(output);

            return output;
        }

        public Vector<double> BackPropagate(Vector<double> truth, Vector<double> test)
        {
            Vector<double> error = LossPrime(truth, test);
            for (int k = Layers.Count - 1; k >= 0; k--)
            {
                error = Layers[k].BackPropagation(error);
            }

            return error;
        }

        public int TrainingFeedback(double lastProgress, int current, int total, double loss)
        {
            int progress = (int)Math.Round(100f * current / total);
            if (lastProgress != progress && progress % 5 == 0)
                Console.Write("[..." + progress + "% | " + "Loss: " + loss + "]");
            return progress;
        }

        public void UpdateParameters(OptimizerType type, int adamIndex, double learningRate)
        {
            Parallel.For(0, ParameterizedLayers.Count, k =>
            {
                ParameterizedLayers[k].UpdateParameters(type, adamIndex, learningRate);
            });
        }

        public void Train(List<Vector<double>> xTrain, List<Vector<double>> yTrain, int epochs, TrainingConfiguration trainingConfiguration = TrainingConfiguration.SGD,
            OptimizerType optimizerType = OptimizerType.Adam, int batchSize = 64, double learningRate = 0.001f)
        {
            switch(trainingConfiguration)
            {
                case TrainingConfiguration.SGD:
                    SGDTrain(xTrain, yTrain, epochs, optimizerType, learningRate);
                    break;
                case TrainingConfiguration.Minibatch:
                    MinibatchTrain(xTrain, yTrain, epochs, optimizerType, batchSize, learningRate);
                    break;
            }
        }

        public void SGDTrain(List<Vector<double>> xTrain, List<Vector<double>> yTrain, int epochs, OptimizerType optimizerType, double learningRate = 0.001f)
        {
            ParameterizedLayers = Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();

            int samples = xTrain.Count;

            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine("\nEpoch: " + (i + 1));

                double err = 0;
                int lastProgress = 0;
                for (int j = 0; j < samples; j++)
                {
                    Vector<double> output = Predict(xTrain[j]);
                    err += Loss(yTrain[j], output);
                    BackPropagate(yTrain[j], output);
                    UpdateParameters(optimizerType, j, learningRate);
                    lastProgress = TrainingFeedback(lastProgress, j, samples, err / j);
                }

                err /= samples;
                Console.WriteLine("\nLoss: " + err + "\n");
            }
        }

        public void MinibatchTrain(List<Vector<double>> xTrain, List<Vector<double>> yTrain, int epochs, OptimizerType optimizerType, int batchSize, double learningRate = 0.001f)
        {
            ParameterizedLayers = Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();
            ParameterizedLayers.ForEach(x => x.SetGradientAccumulation(true));

            var dataTuples = new List<(Vector<double>, Vector<double>)>();
            for (int i = 0; i < xTrain.Count; i++)
                dataTuples.Add((xTrain[i], yTrain[i]));

            Random rnd = new Random();
            var data = dataTuples.OrderBy(x => rnd.Next()).ToList();

            int batchCount = xTrain.Count / batchSize;

            for (int i = 0; i < epochs; i++)
            {
                Console.WriteLine("\nEpoch: " + (i + 1));

                double err = 0;
                int errCount = 0;
                int lastProgress = 0;
                
                for (int b = 0; b <= batchCount; b++)
                {
                    var minibatch = data.Skip(b * batchSize).Take(batchSize).ToList();

                    for(int j = 0; j < minibatch.Count; j++)
                    {
                        Vector<double> output = Predict(minibatch[j].Item1);
                        err += Loss(minibatch[j].Item2, output);
                        BackPropagate(minibatch[j].Item2, output);
                        errCount++;
                    }
                    
                    UpdateParameters(optimizerType, b, learningRate);
                    lastProgress = TrainingFeedback(lastProgress, b, batchCount, err / (b + b * minibatch.Count));
                }

                err /= errCount;
                Console.WriteLine("Loss: " + err + "\n");
            }
        }
        
        //todo: implement batch train
        //todo: write unit tests for all 3 training types

        public string SerializeToJSON()
        {
            return JsonConvert.SerializeObject(this, new JsonSerializerSettings 
            {
                ReferenceLoopHandling = ReferenceLoopHandling.Ignore
            });
        }

        public static Network DeserializeNetworkJSON(string json)
        {
            return JsonConvert.DeserializeObject<Network>(json, new JsonModelParser());
        }
    }
}
