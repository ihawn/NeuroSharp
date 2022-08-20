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
        
        [JsonIgnore]
        public List<ParameterizedLayer> ParameterizedLayers { get; set; }
        
        [JsonIgnore]
        public Func<Vector<double>, Vector<double>, double> Loss { get; set; }
        [JsonIgnore]
        public Func<Vector<double>, Vector<double>, Vector<double>> LossPrime { get; set; }
        
        public LossType LossType { get; set; }
        public string Name { get; set; }

        public Network(string name = "")
        {
            Layers = new List<Layer>();
            Name = name;
        }
        
        //json constructor
        public Network(List<Layer> layers, LossType lossType, string name)
        {
            Layers = layers;
            Name = name;
            UseLoss(lossType);
        }

        public void Add(Layer layer)
        {
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
            }
        }

        public Vector<double> Predict(Vector<double> inputData)
        {
            Vector<double> output = inputData;
            foreach (var layer in Layers)
                output = layer.ForwardPropagation(output);

            return output;
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
                    Vector<double> output = xTrain[j];
                    foreach (var layer in Layers)
                        output = layer.ForwardPropagation(output);

                    err += Loss(yTrain[j], output);

                    // backpropagate the backwards gradient and store weight/bias gradients
                    Vector<double> error = LossPrime(yTrain[j], output);
                    for (int k = Layers.Count - 1; k >= 0; k--)
                    {
                        error = Layers[k].BackPropagation(error);
                    }

                    // update weights/biases based on stored gradients
                    Parallel.For(0, ParameterizedLayers.Count, k =>
                    {
                        ParameterizedLayers[k].UpdateParameters(optimizerType, j, learningRate);
                    });

                    int progress = (int)Math.Round(100f * j / samples);
                    if (lastProgress != progress && progress % 5 == 0)
                        Console.Write("..." + progress + "%");
                    lastProgress = progress;
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

                int lastProgress = 0;
                for (int b = 0; b <= batchCount; b++)
                {
                    var minibatch = data.Skip(b * batchSize).Take(batchSize).ToList();

                    for (int j = 0; j < minibatch.Count; j++)
                    {
                        Vector<double> xTrainItem = minibatch[j].Item1;
                        Vector<double> yTrainItem = minibatch[j].Item2;
                        Vector<double> output = xTrainItem;

                        foreach (var layer in Layers)
                            output = layer.ForwardPropagation(output);

                        err += Loss(yTrainItem, output);

                        // backpropagate the backwards gradient and store weight/bias gradients
                        Vector<double> error = LossPrime(yTrainItem, output);
                        for (int k = Layers.Count - 1; k >= 0; k--)
                        {
                            error = Layers[k].BackPropagation(error);
                        }
                    }

                    int progress = (int)Math.Round(100f * b / batchCount);
                    if (lastProgress != progress && progress % 5 == 0)
                        Console.Write("..." + progress + "%");
                    lastProgress = progress;

                    // update weights/biases based on stored gradients
                    Parallel.For(0, ParameterizedLayers.Count, k =>
                    {
                        ParameterizedLayers[k].UpdateParameters(optimizerType, b, learningRate);
                    });
                }

                err /= batchSize * batchCount;
                Console.WriteLine("Loss: " + err + "\n");
            }
        }

        public string SerializeToJSON()
        {
            return JsonConvert.SerializeObject(this, new JsonSerializerSettings 
            {
                ReferenceLoopHandling = ReferenceLoopHandling.Ignore
            });
        }

        public static Network DeserializeNetworkJSON(string json)
        {
            return JsonConvert.DeserializeObject<Network>(json, new JsonUtils());
        }
    }
}
