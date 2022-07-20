using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System;

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

                Assert.IsTrue(Math.Abs(output - category) < 0.0001d);
            }
        }

        [Test]
        public void ConvolutionOnlyNetwork_ConvolutionLayerNetworkOnlyIsLearning_2DataPieces_1Filter_NoSlide_Stride1()
        {
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

                Assert.IsTrue(Math.Abs(output1 - category1) < 0.1d);
                Assert.IsTrue(Math.Abs(output2 - category2) < 0.1d);
            }
        }
    }
}