using System.Data;
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
            //SentimentAnalysisTraining(12, trainingSize: 2500, testSize: 3000, maxWordCount: 100, maxReviewLength: 50);
        }

        static void LetterIdentificationTraining(int epochs)
        {
            string path = @"C:\Users\Isaac\Documents\C#\NeuroSharp\Data\CharacterRecognition";
            string possibleChars = "abcdefghijklmno";
            
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

            double trainSplit = 1;

            List<Vector<double>> xTrain = 
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTrain = 
                data.Take((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();
            
            List<Vector<double>> xTest = 
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(x => x.Item1).ToList();
            List<Vector<double>> yTest = 
                data.Skip((int)Math.Round(trainSplit * data.Count)).Select(y => y.Item2).ToList();

            Network network = new Network(xTrain[0].Count);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 128, stride: 1, channels: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 64, stride: 1, channels: 128));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 16, stride: 2, channels: 64));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(512));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(yTrain[0].Count));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"C:\Users\Isaac\Documents\C#\NeuroSharp\NeurosharpBlazorWASMServer\NetworkModels\characters_model.json", modelJson);

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

        static void SentimentAnalysisTraining(int epochs, int trainingSize, int testSize, int maxWordCount, int maxReviewLength)
        {
            Random rand = new Random();
            
            string dataPath = @"C:\Users\Isaac\Desktop\reviews\train.csv";
            DataTable dataTable = DataTablePreprocessor.GetDataTableFromCSV(dataPath);

            string allowedChars = "abcdefghijklmnopqrstuvwxyz ";
            List<string> reviews = new List<string>();
            List<int> ratings = new List<int>();
            List<int> order = new List<int>();
            for(int n = 0; n < dataTable.Rows.Count; n++)
                order.Add(n);
            order = order.OrderBy(a => new Random().Next()).ToList();
            foreach(int i in order)
            {
                reviews.Add(new string(dataTable.Rows[i].ItemArray[1].ToString().ToLower()
                    .Where(c => allowedChars.Contains(c)).ToArray()));
                ratings.Add(Int32.Parse(dataTable.Rows[i].ItemArray[0].ToString()));
            }
            
            List<string> allWords = reviews.Select(r => r.Split(' ')
                .ToList()).SelectMany(x => x).ToList();
            List<string> distinctWords = allWords.Distinct().ToList();
            Dictionary<string, int> wordFrequency = new Dictionary<string, int>();

            foreach(string word in allWords)
            {
                if (wordFrequency.ContainsKey(word))
                    wordFrequency[word]++;
                else
                    wordFrequency[word] = 1;
            }

            List<string> uniqueWordsUsedFrequently = wordFrequency.Select(w => new
            {
                word = w.Key,
                wordFrequency = w.Value
            }).OrderByDescending(x => x.wordFrequency).Select(y => y.word)
                .Take(maxWordCount).ToList();
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();
            
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            List<List<string>> processedReviews = new List<List<string>>();

            
            
            for (int i = 0; i < trainingSize + testSize; i++) //todo: write preprocessing classes that will make this easier
            {
                Vector<double> x = Vector<double>.Build.Dense(maxWordCount * maxReviewLength);
                Vector<double> y = Vector<double>.Build.Dense(2);

                List<string> reviewWords = reviews[i].Split(' ')
                    .Where(s => uniqueWordsUsedFrequently.Contains(s)).ToList();
                processedReviews.Add(reviewWords);
                
                for (int j = 0; j < Math.Min(reviewWords.Count, maxReviewLength); j++)
                {
                    int wordIndex = uniqueWordsUsedFrequently.IndexOf(reviewWords[j]);
                    x[j * maxWordCount + wordIndex] = 1;
                }

                y[ratings[i] - 1] = 1;

                if (i < trainingSize)
                {
                    xTrain.Add(x);
                    yTrain.Add(y);
                }
                else
                {
                    xTest.Add(x);
                    yTest.Add(y);
                }
            }
            
            

            Network network = new Network(maxWordCount * maxReviewLength);
            network.Add(new LSTMLayer(maxWordCount, 128, maxReviewLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);
            
            network.Train(xTrain, yTrain, epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, learningRate: 0.018);
            
            string modelJson = network.SerializeToJSON();
            File.WriteAllText(@"C:\Users\Isaac\Documents\C#\NeuroSharp\NeurosharpBlazorWASMServer\NetworkModels\sentiment_analysis_model.json", modelJson);

            int wrongCount = 0;
            for (int i = 0; i < xTest.Count; i++)
            {
                Vector<double> pred = network.Predict(xTest[i]);
                int predNum = pred.ToList().IndexOf(pred.Max()) + 1;
                int actualNum = yTest[i].ToList().IndexOf(1) + 1;
                Console.WriteLine("Prediction: " + predNum);
                Console.WriteLine("Actual: " + actualNum);

                if (predNum != actualNum)
                    wrongCount++;
            }

            double perc = 1 - (double)wrongCount / xTest.Count;
            Console.WriteLine("Accuracy: " + perc);
        }
    }
}