using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Utilities;
using NeuroSharp.Data;
using NeuroSharp.Enumerations;
using NeuroSharp.Datatypes;
using MathNet.Numerics;
using NeuroSharp.Training;

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
            //Mnist_Digits_Test(6000, 100, 5, "digits");
            //Mnist_Digits_Test_Conv(6000, 100, 5, "digits");
            //Conv_Base_Test(1000, 100, 10, "digits");
            //Conv_Vs_Non_Conv(5000, 1000, 15, 20, "digits");
            IntelImageClassification_Conv(epochs: 1);
            //IntelImageClassification_Dense(epochs: 1);
            //BirdSpecies_Test(epochs: 5);

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
            network.Add(new FullyConnectedLayer(256));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(128));
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
            ImageDataAggregate trainData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification_Smaller\seg_train\seg_train", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);
            ImageDataAggregate testData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification_Smaller\seg_test\seg_test", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);

            Network network = new Network(150 * 150 * 3);
            network.Add(new ConvolutionalLayer(kernel: 3, filters: 16, stride: 3, channels: 3));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MaxPoolingLayer(poolSize: 3));
            network.Add(new ConvolutionalLayer(kernel: 2, filters: 4, stride: 2, channels: 16));
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

            string modelJson = network.SerializeToJSON();
            System.IO.File.WriteAllText (@"C:\Users\Isaac\Desktop\intel_image_classification_model.txt", modelJson);
        }
        static void IntelImageClassification_Dense(int epochs)
        {
            ImageDataAggregate trainData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification_Smaller\seg_train\seg_train", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);
            ImageDataAggregate testData = ImagePreprocessor.GetImageData(@"C:\Users\Isaac\Desktop\IntelImageClassification_Smaller\seg_test\seg_test", ImagePreprocessingType.ParentFolderContainsLabel, expectedHeight: 150, expectedWidth: 150);

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

            string modelJson = network.SerializeToJSON();
            System.IO.File.WriteAllText (@"C:\Users\Isaac\Desktop\intel_image_classification_model.txt", modelJson);
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
