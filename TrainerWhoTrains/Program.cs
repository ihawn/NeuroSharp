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
            LetterIdentificationTraining(20);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            string path = @"C:\Users\Isaac\Downloads";
            string possibleChars = "abcdef";
            
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();
           
            foreach (string file in Directory.EnumerateFiles(path, "*.txt"))
            {
                Vector<double> x = Vector<double>.Build.DenseOfEnumerable(
                   File.ReadAllText(file).Split(",").Select(x => double.Parse(x))
                );
                
                string character = file.Replace(path + @"\", "")[0].ToString();
                Vector<double> y = Vector<double>.Build.Dense(possibleChars.Length);
                y[possibleChars.IndexOf(character)] = 1;
                
                xTrain.Add(x);
                yTrain.Add(y);
            }

            Network network = new Network(xTrain[0].Count);
            /*network.Add(new ConvolutionalLayer(kernel: 2, filters: 16, stride: 1, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));*/
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 16, stride: 2, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 4, stride: 1, channels: 16));
            network.Add(new ActivationLayer(ActivationType.ReLu));
           /* network.Add(new FullyConnectedLayer(512));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));*/
            network.Add(new FullyConnectedLayer(196));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(yTrain[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"C:\Users\Isaac\Documents\C#\NeuroSharp\NeurosharpBlazorWASM\wwwroot\NetworkModels\characters_model.json", modelJson);
            
           /* int i = 0;
            int wrongCount = 0;
            foreach(var test in xTest)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if(prediction != actual)
                    wrongCount++;

                i++;
            }
            double acc = (1d - ((double)wrongCount) / i);
            Console.WriteLine("Accuracy: " + acc);*/
        }
    }
}