using MathNet.Numerics.LinearAlgebra;
using NeuroSharp;
using NeuroSharp.Data;
using NeuroSharp.Datatypes;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;

namespace Trainer
{
    public class Trainer
    {
        static void Main(string[] args)
        {
            LetterIdentificationTraining(15);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            string path = @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\CharacterRecognition";
            string possibleChars = "abcdef";
            
            List<(Vector<double>, Vector<double>)> data = new List<(Vector<double>, Vector<double>)>();
           
            foreach (string file in Directory.EnumerateFiles(path, "*.txt"))
            {
                Vector<double> x = Vector<double>.Build.DenseOfEnumerable(
                   File.ReadAllText(file).Split(",").Select(x => double.Parse(x))
                );
                
                string character = file.Replace(path + @"\", "")[0].ToString();
                Vector<double> y = Vector<double>.Build.Dense(possibleChars.Length);
                y[possibleChars.IndexOf(character)] = 1;
                
                data.Add((x, y));
            }

            Random rand = new Random();
            data = data.OrderBy(x => rand.Next()).ToList();

            double trainSplit = 0.9;

            List<Vector<double>> xTrain = 
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTrain = 
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();
            
            List<Vector<double>> xTest = 
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTest = 
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();

            Network network = new Network(xTrain[0].Count);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 40, stride: 1, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 12, stride: 2, channels: 40));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(yTrain[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"C:\Users\Isaac\Documents\C#\NeuroSharp\NeurosharpBlazorWASM\wwwroot\NetworkModels\characters_model.json", modelJson);

            if (trainSplit < 1)
            {
                int i = 0;
                int wrongCount = 0;
                foreach (var test in xTest)
                {
                    var output = network.Predict(test);
                    int prediction = output.ToList().IndexOf(output.Max());
                    int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                    Console.WriteLine("Prediction: " + prediction);
                    Console.WriteLine("Actual: " + actual + "\n");

                    if (prediction != actual)
                        wrongCount++;

                    i++;
                }

                double acc = (1d - ((double)wrongCount) / i);
                Console.WriteLine("Accuracy: " + acc);
            }
        }
    }
}