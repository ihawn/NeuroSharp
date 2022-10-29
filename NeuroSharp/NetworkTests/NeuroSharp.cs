using System.Data;
using System.Text.RegularExpressions;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Utilities;
using NeuroSharp.Data;
using NeuroSharp.Enumerations;
using NeuroSharp.Datatypes;
using MathNet.Numerics;
using NeuroSharp.Training;
using Microsoft.VisualBasic.FileIO;

namespace NeuroSharp
{
    public class NeuroSharp
    {
        static void Main(string[] args)
        {
            //Control.UseNativeCUDA();
            Control.UseNativeMKL();
            //Control.UseManaged();

            //XOR_Test();
            //Mnist_Digits_Test(600, 100, 5, "digits");
            //Mnist_Digits_Test_Conv(6000, 100, 5, "digits");
            //Mnist_Digits_Test_Binary(60000, 10000, 5, "digits");
            //Conv_Base_Test(1000, 100, 10, "digits");
            //Conv_Vs_Non_Conv(5000, 1000, 15, 20, "digits");
            //IntelImageClassification_Conv(epochs: 5);
            //IntelImageClassification_Dense(epochs: 5);
            //BirdSpecies_Test(epochs: 5);
            //RNN_Sentence_Prediction(101, 20, 5, 20);
            //RNN_Name_Generator(epochs: 5, promptLength: 2, nameLengthAfterPrompt: 15, trainingSize: 20000);
            //RNN_Sequence_Prediction(20000, 10, 5, 3);
            RNN_Sentiment_Analysis(20, 3000, 300); //todo: dimension transform layer

            #region testing

            /* // Using managed code only
             var m1 = Matrix<double>.Build.Random(10000, 10000);
             var m2 = Matrix<double>.Build.Random(10000, 10000);
             var w = Stopwatch.StartNew();

             Control.UseManaged();
             Console.WriteLine("Managed");

             var y1 = m1 * m2;
             Console.WriteLine(w.Elapsed);
             Console.WriteLine(y1);

             // Using the Intel MKL native provider
             Control.UseNativeMKL();
             Console.WriteLine("MKL");

             w.Restart();
             var y2 = m1 * m2;
             Console.WriteLine(w.Elapsed);
             Console.WriteLine(y2);

             // Cuda ??
             /*Control.UseNativeCUDA();
             Console.WriteLine("CUDA");

             w.Restart();
             var y3 = m1 * m2;
             Console.WriteLine(w.Elapsed);
             Console.WriteLine(y3);*/

            /*var m1 = Matrix<double>.Build.Random(20000, 20000);
            var w = Stopwatch.StartNew();

            Control.UseManaged();
            Console.WriteLine("Managed");

            var y1 = m1.Multiply(0.2d);
            Console.WriteLine(w.Elapsed);
            Console.WriteLine(y1);

            // Using the Intel MKL native provider
            Control.UseNativeMKL();
            Console.WriteLine("MKL");

            w.Restart();
            var y2 = m1.Multiply(0.2d);
            Console.WriteLine(w.Elapsed);
            Console.WriteLine(y2);*/

            #endregion
        }

        static void XOR_Test()
        {
            double[][] xx =
            {
                new double[]{ 0, 0 },
                new double[]{ 0, 1 },
                new double[]{ 1, 0 },
                new double[]{ 1, 1 }
            };
            List<Vector<double>> xTrain = new List<Vector<double>>();
            foreach (var x in xx)
                xTrain.Add(Vector<double>.Build.DenseOfArray(x));

            double[][] yy =
            {
                new double[]{ 0 },
                new double[]{ 1 },
                new double[]{ 1 },
                new double[]{ 0 }
            };
            List<Vector<double>> yTrain = new List<Vector<double>>();
            foreach (var y in yy)
                yTrain.Add(Vector<double>.Build.DenseOfArray(y));


            Network network = new Network(2);
            network.Add(new FullyConnectedLayer(3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(1));
            network.UseLoss(LossType.MeanSquaredError);

            network.SGDTrain(xTrain, yTrain, epochs: 1000, optimizerType: OptimizerType.GradientDescent, learningRate: 0.1f);

            //test
            foreach(var test in xTrain)
            {
                var output = network.Predict(test);
                foreach (var o in output)
                    Console.WriteLine((double)o);
            }

            string modelJson = network.SerializeToJSON();
        }
        static double Mnist_Digits_Test(int trainSize, int testSize, int epochs, string data)
        {
            //training data
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            var trainData = MnistReader.ReadTrainingData(data).ToList();
            for(int n = 0; n < trainSize; n++)
            {
                var image = trainData[n];

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t/256d).ToArray();
                xTrain.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[10];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            //testing data
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            var testData = MnistReader.ReadTestData(data).ToList();
            for (int n = 0; n < testSize; n++)
            {
                var image = testData[n];

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTest.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[10];
                categorical[image.Label] = 1;
                yTest.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            //build network
            Network network = new Network(28 * 28);
            network.Add(new FullyConnectedLayer(200));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(10));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.001f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
            int i = 0;
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
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");

            string networkJson = network.SerializeToJSON();
            
            return acc;
        }
        static double Mnist_Digits_Test_Binary(int trainSize, int testSize, int epochs, string data)
        {
            //training data
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            var trainData = MnistReader.ReadTrainingData(data).ToList();
            for(int n = 0; n < trainSize; n++)
            {
                var image = trainData[n];
                if(image.Label > 1)
                    continue;

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t/256d).ToArray();
                xTrain.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[2];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            //testing data
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            var testData = MnistReader.ReadTestData(data).ToList();
            for (int n = 0; n < testSize; n++)
            {
                var image = testData[n];
                if(image.Label > 1)
                    continue;

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTest.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[2];
                categorical[image.Label] = 1;
                yTest.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            //build network
            Network network = new Network(28 * 28);
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.Train(xTrain, yTrain, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.001f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
            int i = 0;
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
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");

            string networkJson = network.SerializeToJSON();
            
            return acc;
        }
        static double Mnist_Digits_Test_Conv(int trainSize, int testSize, int epochs, string data)
        {
            //training data
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            var trainData = MnistReader.ReadTrainingData(data).ToList();
            for (int n = 0; n < trainSize; n++)
            {
                var image = trainData[n];

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTrain.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[10];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<double>.Build.DenseOfArray(categorical));
            }


            //testing data
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            var testData = MnistReader.ReadTestData(data).ToList();
            for (int n = 0; n < testSize; n++)
            {
                var image = testData[n];

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTest.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[10];
                categorical[image.Label] = 1;
                yTest.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            /* for(int n = 0; n < xTrain.Count; n++)
                 xTrain[n] = PCA.GetPrincipleComponents(xTrain[n], 28);
             for (int n = 0; n < xTest.Count; n++)
                 xTest[n] = PCA.GetPrincipleComponents(xTest[n], 28);*/

            //build network
            Network network = new Network(28 * 28);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 2, channels: 1, stride: 2));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(10));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.Train(xTrain, yTrain, epochs: epochs, trainingConfiguration: TrainingConfiguration.SGD, OptimizerType.Adam, batchSize: 64, learningRate: 0.001f);
            var elapsedMs = watch.ElapsedMilliseconds;

            string model = network.SerializeToJSON();
            Network deserializedNetwork = Network.DeserializeNetworkJSON(model);

            //test
            int i = 0;
            int wrongCount = 0;
            foreach (var test in xTest)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                if (prediction != actual)
                    wrongCount++;
                i++;
            }
            
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);

            i = 0;
            wrongCount = 0;
            foreach (var test in xTest)
            {
                var output = deserializedNetwork.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                if (prediction != actual)
                    wrongCount++;
                i++;
            }
            
            acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy from deserialized network: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");
            
            string modelJson = network.SerializeToJSON();
            
            return acc;
        }
        static void RNN_Sentence_Prediction(int trainSize, int testSize, int wordsAfterPrompt, int epochs)
        {
            Random rand = new Random();
            
            string text = File.ReadAllText(@"C:\Users\Isaac\Desktop\short_stories.txt");
            string allowedChars = "abcdefghijklmnopqrstuvwxyz. ";
            
            text = text.ToLower();
            text = Regex.Replace(text, @"[\r\n]+", " ");
            text = new string(text.Where(c => allowedChars.Contains(c)).ToArray());
            
            List<string> sentences = text.Split(". ").Where(s => s.Split(" ").ToList().Count() >= 2)
                .OrderBy(x => rand.Next()).ToList();
            List<string> uniqueWords = (new string(text.Where(c => c != '.').ToArray())).Split(" ")
                .Distinct().Where(x => x.Any()).OrderBy(y => y).ToList();
            
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();
            
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            for (int i = 0; i < trainSize + testSize; i++)
            {
                Vector<double> x = Vector<double>.Build.Dense(uniqueWords.Count);
                List<string> sentenceWords = sentences[i].Split(" ").Where(t => t != "").ToList();
                int wordIndex = uniqueWords.IndexOf(sentenceWords[0]);
                if(wordIndex == -1) continue;
                x[wordIndex] = 1;

                Vector<double> y = Vector<double>.Build.Dense(uniqueWords.Count);
                sentenceWords = sentences[i].Split(" ").Where(t => t != "").ToList();
                wordIndex = uniqueWords.IndexOf(sentenceWords[1]);
                if(wordIndex == -1) continue;
                y[wordIndex] = 1;

                if (i < trainSize)
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

            Network network = new Network(uniqueWords.Count);
            network.Add(new RecurrentLayer(1, uniqueWords.Count, 128));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.Train(xTrain, yTrain, epochs, TrainingConfiguration.SGD, OptimizerType.GradientDescent, learningRate: 0.03);
            
            foreach(var test in xTest)
            {
                Vector<double> predictionVector = network.Predict(test);
                int predictionIndex = predictionVector.ToList().IndexOf(predictionVector.Max());
                int promptIndex = test.ToList().IndexOf(test.Max());
                Console.WriteLine(uniqueWords[promptIndex] + " " + uniqueWords[predictionIndex]);
            }
        }
        static void RNN_Sequence_Prediction(int trainSize, int testSize, int sequenceLength, int epochs)
        {
            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();
            
            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            Random rand = new Random();
            double step = Math.PI / 8;
            
            for (int i = 0; i < trainSize + testSize; i++)
            {
                Vector<double> x = Vector<double>.Build.Dense(sequenceLength);
                Vector<double> y = Vector<double>.Build.Dense(sequenceLength);

                double start = rand.NextDouble() * 10 - 5;
                for(int j = 0; j < sequenceLength; j++)
                {
                    x[j] = Math.Sin(start + j * step);
                    y[j] = Math.Sin(start + sequenceLength * step + j * step);
                }

                if (i < trainSize)
                {
                    xTrain.Add(x);
                    yTrain.Add(y);
                }
                else
                {
                    xTest.Add(x);
                    yTest.Add(x);
                }
            }


            Network network = new Network(sequenceLength);
            network.Add(new RecurrentLayer(sequenceLength, 1, 64));
            network.UseLoss(LossType.MeanSquaredError);
            
            network.Train(xTrain, yTrain, epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002);

            for(int i = 0; i < testSize; i++)
            {
                Vector<double> prediction = network.Predict(xTest[i]);
                Console.WriteLine("Prediction distance: " + (prediction - yTest[i]).L2Norm());
            }
        }
        static void RNN_Name_Generator(int epochs, int promptLength, int nameLengthAfterPrompt, int trainingSize)
        {
            Random rand = new Random();

            string letters = "abcdefghijklmnopqr.stuvwxyz";

            List<string> text = File.ReadAllText(@"C:\Users\Isaac\Desktop\first-names.txt")
                .ToLower().Split("\n").OrderBy(x => rand.Next()).Take(trainingSize).ToList();
            for (int i = 0; i < text.Count; i++)
            {
                text[i] = Regex.Replace(text[i], @"[\r\n]+", "");
                text[i] = new string(text[i].Where(c => letters.Contains(c)).ToArray());
                text[i] += ".";
            }

            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            for (int i = 0; i < text.Count; i++)
            {
                if(text[i].Length < promptLength + 1)
                    continue;

                int l = text[i].Length;
                int start = rand.Next(0, l - promptLength);
                string wordData = text[i].Substring(start, promptLength + 1);
                
                double[] name = new double[27 * promptLength];
                for(int j = 0; j < promptLength; j++)
                    name[27 * j + letters.IndexOf(wordData[j])] = 1;
                xTrain.Add(Vector<double>.Build.DenseOfArray(name));
                
                double[] nextLetter = new double[27];
                nextLetter[letters.IndexOf(wordData[wordData.Length - 1])] = 1;
                yTrain.Add(Vector<double>.Build.DenseOfArray(nextLetter));
            }

            Network network = new Network(27 * promptLength);
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(27 * promptLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new RecurrentLayer(promptLength, 27, 128));
            network.Add(new RecurrentLayer(promptLength, 27, 128));
            network.Add(new FullyConnectedLayer(27));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.Train(xTrain, yTrain, epochs, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.001);

            List<string> prompts = new List<string>();
            for (int i = 0; i < letters.Length; i++)
            {
                if(letters[i] == '.') continue;
                for (int j = 0; j < letters.Length; j++)
                {
                    if(letters[j] == '.') continue;
                    prompts.Add(letters[i].ToString() + letters[j].ToString());
                }
            }
            foreach (string s in prompts)
            {
                string newName = s;
                for (int i = 0; i < nameLengthAfterPrompt; i++)
                {
                    string subPrompt = newName.Substring(newName.Length - promptLength);
                    Vector<double> xTest = Vector<double>.Build.Dense(27 * promptLength);
                    for(int j = i; j < promptLength + i; j++)
                        xTest[27 * (j - i) + letters.IndexOf(subPrompt[j - i])] = 1;
                    
                    Vector<double> pred = network.Predict(xTest);
                    char nextChar = letters[pred.ToList().IndexOf(pred.Max())];
                    if (nextChar == '.')
                        break;
                    newName += nextChar;
                }

                Console.WriteLine(newName);
            }
        }
        static void RNN_Sentiment_Analysis(int epochs, int trainingSize, int testSize)
        {
            Random rand = new Random();
            
            ///
            string dataPath = @"C:\Users\Isaac\Desktop\reviews\train_shorter.csv";
            DataTable dataTable = DataTablePreprocessor.GetDataTableFromCSV(dataPath);

            string allowedChars = "abcdefghijklmnopqrstuvwxyz ";
            List<string> reviews = new List<string>();
            List<int> ratings = new List<int>();
            for (int i = 0; i < dataTable.Rows.Count; i++)
            {
                reviews.Add(new string(dataTable.Rows[i].ItemArray[1].ToString().ToLower()
                    .Where(c => allowedChars.Contains(c)).ToArray()));
                ratings.Add(Int32.Parse(dataTable.Rows[i].ItemArray[0].ToString()));
            }
            ///

            ///
            int maxWordCount = 32;
            int maxReviewLength = 25;
            
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
            
            
            ///
            Network network = new Network(maxWordCount * maxReviewLength);
            network.Add(new LongShortTermMemoryLayer(maxWordCount, 96, maxReviewLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);
            
            network.Train(xTrain, yTrain, epochs, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.003);

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
        static void Conv_Vs_Non_Conv(int trainSize, int testSize, int testsToRun, int epochs, string data)
        {
            double denseNetAcc = 0;
            double convNetAcc = 0;

            for(int i = 0; i < testsToRun; i++)
            {
                denseNetAcc += Mnist_Digits_Test(trainSize, testSize, epochs, data);
                convNetAcc += Mnist_Digits_Test_Conv(trainSize, testSize, epochs, data);
            }

            denseNetAcc /= testsToRun;
            convNetAcc /= testsToRun;
            
            Console.WriteLine("Dense Network Average Accuracy: " + denseNetAcc);
            Console.WriteLine("Convolutional Network Average Accuracy: " + convNetAcc);
        }
        static void IntelImageClassification_Conv(int epochs)
        {
            ImageDataAggregate trainData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification\seg_train\seg_train", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);
            ImageDataAggregate testData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification\seg_test\seg_test", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);

            Network network = new Network(150 * 150 * 3);
            network.Add(new ConvolutionalLayer(kernel: 3, filters: 8, stride: 3, channels: 3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 3));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 4, stride: 2, channels: 8));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(6));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.Train(trainData.XValues, trainData.YValues, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
            string[] labels = new string[] { "buildings", "forest", "glacier", "mountain", "sea", "street" };
            int i = 0;
            int wrongCount = 0;
            foreach (var test in testData.XValues)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = testData.YValues[i].ToList().IndexOf(testData.YValues[i].Max());
                Console.WriteLine("Prediction: " + labels[prediction]);
                Console.WriteLine("Actual: " + labels[actual] + "\n");

                if (prediction != actual)
                    wrongCount++;

                i++;
            }
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");

            //string modelJson = network.SerializeToJSON();
            //System.IO.File.WriteAllText (@"C:\Users\Isaac\Desktop\intel_image_classification_model.txt", modelJson);
        }
        static void IntelImageClassification_Dense(int epochs)
        {
            ImageDataAggregate trainData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification\seg_train\seg_train", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);
            ImageDataAggregate testData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification\seg_test\seg_test", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);

            Network network = new Network(150 * 150 * 3);
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(128));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(64));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(6));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.Train(trainData.XValues, trainData.YValues, epochs: epochs, TrainingConfiguration.Minibatch, OptimizerType.Adam, batchSize: 64, learningRate: 0.002f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
            string[] labels = new string[] { "buildings", "forest", "glacier", "mountain", "sea", "street" };
            int i = 0;
            int wrongCount = 0;
            foreach (var test in testData.XValues)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = testData.YValues[i].ToList().IndexOf(testData.YValues[i].Max());
                Console.WriteLine("Prediction: " + labels[prediction]);
                Console.WriteLine("Actual: " + labels[actual] + "\n");

                if (prediction != actual)
                    wrongCount++;

                i++;
            }
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");

            //string modelJson = network.SerializeToJSON();
            //System.IO.File.WriteAllText (@"C:\Users\Isaac\Desktop\intel_image_classification_model.txt", modelJson);
        }
        static void BirdSpecies_Test(int epochs)
        {
            ImageDataAggregate trainData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\Birds\train", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 224, expectedWidth: 224);
            ImageDataAggregate testData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\Birds\test", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 224, expectedWidth: 224);

            Network network = new Network(224 * 224 * 3);
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 4, stride: 3, channels: 3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 3));
            network.Add(new ConvolutionalLayer(kernel: 3, filters: 3, stride: 2, channels: 4));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 3));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 2, stride: 2, channels: 3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 3));
            network.Add(new ConvolutionalLayer(kernel: 3, filters: 1, stride: 1, channels: 2));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 2));
            network.Add(new FullyConnectedLayer(37));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            network.SGDTrain(trainData.XValues, trainData.YValues, epochs: epochs, OptimizerType.Adam, learningRate: 0.0001f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
            int i = 0;
            int wrongCount = 0;
            foreach (var test in testData.XValues)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = testData.YValues[i].ToList().IndexOf(testData.YValues[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if (prediction != actual)
                    wrongCount++;

                i++;
            }
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");
        }
    }
}
