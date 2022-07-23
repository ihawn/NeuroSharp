using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Data;

namespace UnitTests
{
    public class ConvolutionalNetworkTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ConvolutionOnlyNetwork_ConvolutionLayerNetworkOnlyIsLearning_1DataPiece_1Filter_NoSlide_Stride1()
        {
            for(int i = 0; i < 50; i++)
            {
                Random rand = new Random();

                double[] x1 = new double[]
                {
                    1, 0, 0,
                    1, 0, 1,
                    1, 0, 0,
                };

                double category = rand.NextDouble();
                double[] y1 = new double[] { category };
                List<Vector<double>> xTrain = new List<Vector<double>> { Vector<double>.Build.DenseOfArray(x1) };
                List<Vector<double>> yTrain = new List<Vector<double>> { Vector<double>.Build.DenseOfArray(y1) };

                Network network = new Network();
                network.Add(new ConvolutionalLayer(3 * 3, kernel: 3, filters: 1, stride: 1));
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                network.Train(xTrain, yTrain, epochs: 500, OptimizerType.Adam, learningRate: 0.005f);

                double output = network.Predict(xTrain[0])[0];

                Console.WriteLine("Prediction: " + output);
                Console.WriteLine("Actual: " + category + "\n");

                Assert.IsTrue(Math.Abs(output - category) < 0.01d);
            }
        }

        [Test]
        public void ConvolutionOnlyNetwork_ConvolutionLayerNetworkOnlyIsLearning_2DataPieces_1Filter_NoSlide_Stride1()
        {
            double correct = 0;
            for (int i = 0; i < 50; i++)
            {
                double[] x1 = new double[]
                {
                    1, 0, 0,
                    1, 0, 1,
                    1, 0, 0,
                };
                double[] x2 = new double[]
                {
                    0, 0, 1,
                    1, 0, 1,
                    0, 0, 1,
                };

                Random rand = new Random();
                double category1 = rand.NextDouble();
                double category2 = rand.NextDouble();
                double[] y1 = new double[] { category1 };
                double[] y2 = new double[] { category2 };

                List<Vector<double>> xTrain = new List<Vector<double>>
                {
                    Vector<double>.Build.DenseOfArray(x1),
                    Vector<double>.Build.DenseOfArray(x2),
                };

                List<Vector<double>> yTrain = new List<Vector<double>>
                {
                    Vector<double>.Build.DenseOfArray(y1),
                    Vector<double>.Build.DenseOfArray(y2),
                };

                Network network = new Network();
                network.Add(new ConvolutionalLayer(3 * 3, kernel: 3, filters: 1, stride: 1));
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                network.Train(xTrain, yTrain, epochs: 500, OptimizerType.Adam, learningRate: 0.005f);

                double output1 = network.Predict(xTrain[0])[0];
                double output2 = network.Predict(xTrain[1])[0];

                Console.WriteLine("Prediction: " + output1);
                Console.WriteLine("Actual: " + category1 + "\n");

                Console.WriteLine("Prediction: " + output2);
                Console.WriteLine("Actual: " + category2 + "\n");

                correct += Math.Abs(output1 - category1) < 0.15 ? 0.5 : 0;
                correct += Math.Abs(output2 - category2) < 0.15 ? 0.5 : 0;
            }

            Assert.IsTrue(correct > 40);
        }

        [Test]
        public void ConvolutionOnlyNetwork_ConvolutionLayerMaxPoolingNetworkOnlyIsLearning_2DataPieces_1Filter_Slide_Stride1()
        {
            int correct = 0;
            for(int i = 0; i < 30; i++)
            {
                double[] x1 = new double[]
                {
                    1, 0, 0,
                    1, 0, 1,
                    1, 0, 0,
                };

                Random rand = new Random();
                double category = rand.NextDouble();
                double[] y1 = new double[] { category };

                List<Vector<double>> xTrain = new List<Vector<double>>
                {
                    Vector<double>.Build.DenseOfArray(x1),
                };

                List<Vector<double>> yTrain = new List<Vector<double>>
                {
                    Vector<double>.Build.DenseOfArray(y1),
                };

                Network network = new Network();
                network.Add(new ConvolutionalLayer(3 * 3, kernel: 2, filters: 1, stride: 1));
                network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
                network.Add(new MaxPoolingLayer(4, prevFilterCount: 1, poolSize: 2));
                network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

                network.Train(xTrain, yTrain, epochs: 2000, OptimizerType.Adam, learningRate: 0.002);

                var output = network.Predict(xTrain[0])[0];
                var actual = category;
                Console.WriteLine("Prediction: " + output);
                Console.WriteLine("Actual: " + actual + "\n");

                correct += Math.Abs(output - actual) < 0.15 ? 1 : 0;
            }

            Assert.IsTrue(correct > 20);
        }

        /*[Test]
        public void ConvolutionWithDense_ShouldBeBetterThanDenseOnly()
        {
            double err1 = 0;
            double err2 = 0;

            List<Vector<double>> xTrain = new List<Vector<double>>();
            List<Vector<double>> yTrain = new List<Vector<double>>();

            var trainData = MnistReader.ReadTrainingData("digits").ToList();
            for (int n = 0; n < 100; n++)
            {
                var image = trainData[n];
                if (image.Label > 1)
                    continue;

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTrain.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[2];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<double>.Build.DenseOfArray(categorical));
            }


            List<Vector<double>> xTest = new List<Vector<double>>();
            List<Vector<double>> yTest = new List<Vector<double>>();

            var testData = MnistReader.ReadTestData("digits").ToList();
            for (int n = 0; n < 100; n++)
            {
                var image = testData[n];
                if (image.Label > 1)
                    continue;

                double[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256d).ToArray();
                xTest.Add(Vector<double>.Build.DenseOfArray(flattenedNormalized));

                double[] categorical = new double[2];
                categorical[image.Label] = 1;
                yTest.Add(Vector<double>.Build.DenseOfArray(categorical));
            }

            //dense layer network only
            Network network1 = new Network();
            network1.Add(new FullyConnectedLayer(28*28, 2));
            network1.Add(new SoftmaxActivationLayer());
            network1.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            network1.Train(xTrain, yTrain, epochs: 10, OptimizerType.Adam, learningRate: 0.002);

            int i = 0;
            foreach (var test in xTrain)
            {
                var output = network1.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTrain[i].ToList().IndexOf(yTrain[i].Max());
                Console.WriteLine("Prediction Vector: " + output);
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");
                err1 += (output - yTrain[i]).L2Norm();
                i++;
            }

            //with conv layer
            Network network2 = new Network();
            network2.Add(new ConvolutionalLayer(28*28, kernel: 3, filters: 1, stride: 1));
            network2.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network2.Add(new FullyConnectedLayer(26*26, 2));
            network2.Add(new SoftmaxActivationLayer());
            network2.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            network2.Train(xTrain, yTrain, epochs: 5, OptimizerType.Adam, learningRate: 0.002);

            i = 0;
            foreach (var test in xTrain)
            {
                var output = network2.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTrain[i].ToList().IndexOf(yTrain[i].Max());
                Console.WriteLine("Prediction Vector: " + output);
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");
                err2 += (output - yTrain[i]).L2Norm();
                i++;
            }

            Assert.IsTrue(err1 > err2);
        }*/
    }
}