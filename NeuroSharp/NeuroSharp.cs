﻿using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.MathUtils;
using NeuroSharp.Data;
using NeuroSharp.Enumerations;
using MathNet.Numerics;
using System;
using System.Diagnostics;

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
            //Mnist_Digits_Test(512, 10, 5, "digits");
            Mnist_Digits_Test_Conv(1024, 100, 5, "digits");
            //Conv_Base_Test(1000, 100, 10, "digits");
            //Conv_Vs_Non_Conv(5000, 1000, 15, 20, "digits");

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


            Network network = new Network();
            network.Add(new FullyConnectedLayer(2, 3));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            //network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(3, 1));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            //network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));

            //train
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
            network.Train(xTrain, yTrain, epochs: 1000, optimizerType: OptimizerType.GradientDescent, learningRate: 0.1f);

            //test
            foreach(var test in xTrain)
            {
                var output = network.Predict(test);
                foreach (var o in output)
                    Console.WriteLine((double)o);
            }
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

           /* for(int n = 0; n < xTrain.Count; n++)
                xTrain[n] = PCA.GetPrincipleComponents(xTrain[n], 28);
            for (int n = 0; n < xTest.Count; n++)
                xTest[n] = PCA.GetPrincipleComponents(xTest[n], 28);*/

            //build network
            Network network = new Network();
            network.Add(new FullyConnectedLayer(28*28, 256));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new FullyConnectedLayer(256, 128));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new FullyConnectedLayer(128, 10));
            //network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            //network.Train(xTrain, yTrain, epochs: 5, OptimizerType.Adam);
            network.MinibatchTrain(xTrain, yTrain, epochs: epochs, OptimizerType.GradientDescent, batchSize: 256);
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
            return acc;
        }
        static void Conv_Base_Test(int trainSize, int testSize, int epochs, string data)
        {
            double[] x1 = new double[]
            {
                1, 0, 0,
                0, 0, 0,
                0, 0, 0
            };
            double[] x2 = new double[]
            {
                0, 0, 1,
                0, 0, 0,
                0, 0, 0
            };
            double[] x3 = new double[]
            {
                0, 0, 0,
                0, 0, 0,
                1, 0, 0
            };
            double[] x4 = new double[]
            {
                0, 0, 0,
                0, 0, 0,
                0, 0, 1,
            };

            /*double[] x1 = new double[]
            {
                1, 0, 0,
                1, 0, 0,
                1, 0, 0,
            };
            double[] x2 = new double[]
            {
                0, 0, 1,
                0, 0, 1,
                0, 0, 1,
            };*/


            double[] y1 = new double[] { 1, 0, 0, 0 };
            double[] y2 = new double[] { 0, 1, 0, 0 };
            double[] y3 = new double[] { 0, 0, 1, 0 };
            double[] y4 = new double[] { 0, 0, 0, 1 };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(x1),
                Vector<double>.Build.DenseOfArray(x2),
                Vector<double>.Build.DenseOfArray(x3),
                Vector<double>.Build.DenseOfArray(x4),
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(y1),
                Vector<double>.Build.DenseOfArray(y2),
                Vector<double>.Build.DenseOfArray(y3),
                Vector<double>.Build.DenseOfArray(y4),
            };

            //build network
            Network network = new Network();
            network.Add(new ConvolutionalLayer(9, kernel: 2, filters: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            //network.Add(new MaxPoolingLayer(4, prevFilterCount: 1, poolSize: 2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);
            //network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            //train
            network.Train(xTrain, yTrain, epochs: 500, OptimizerType.Adam, learningRate: 0.001);

            //test
            int i = 0;
            foreach (var test in xTrain)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTrain[i].ToList().IndexOf(yTrain[i].Max());
                Console.WriteLine("Prediction Vector: " + output);
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");
                i++;
            }
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
            Network network = new Network();
            network.Add(new ConvolutionalLayer(28 * 28, kernel: 2, filters: 2, stride: 1));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new MaxPoolingLayer(27 * 27 * 2, prevFilterCount: 2, poolSize: 2));
            network.Add(new FullyConnectedLayer(26 * 26 * 2, 128));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(128, 10));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            //train
            var watch = System.Diagnostics.Stopwatch.StartNew();
            //network.Train(xTrain, yTrain, epochs: epochs, OptimizerType.Adam);
            network.MinibatchTrain(xTrain, yTrain, epochs: epochs, OptimizerType.Adam, batchSize: 32, learningRate: 0.001f);
            var elapsedMs = watch.ElapsedMilliseconds;

            //test
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
            double acc = (1f - ((double)wrongCount) / ((double)i));
            Console.WriteLine("Accuracy: " + acc);
            Console.WriteLine("Training Runtime: " + (elapsedMs / 1000f).ToString() + "s");
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
    }
}
