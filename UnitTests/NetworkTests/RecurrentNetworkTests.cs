using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Data;
using NeuroSharp.Training;

namespace UnitTests
{
    public class RecurrentNetworkTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void RNN_LearnsCorrectly_WhenSameDataWithDifferentOrder()
        {
            // vocabSize = 3, sequenceLength = 3
            // john eats apples / apples eat john
            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                    }
                ),
                Vector<double>.Build.DenseOfArray(
                    new double[]
                    {
                        0, 0, 1,
                        0, 1, 0,
                        1, 0, 0,
                    }
                )
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(
                    new double [] { 1, 0 }
                ),
                Vector<double>.Build.DenseOfArray(
                    new double [] { 0, 1 }
                )
            };


            Network network = new Network(9);
            network.Add(new RecurrentLayer(3, 3, 16));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new FullyConnectedLayer(2));
            network.Add(new NeuroSharp.SoftmaxActivationLayer());
            network.UseLoss(LossType.BinaryCrossentropy);
            
            network.Train(xTrain, yTrain, 30, TrainingConfiguration.SGD, OptimizerType.Adam, learningRate: 0.1);

            for (int i = 0; i < xTrain.Count; i++)
            {
                Vector<double> predVector = network.Predict(xTrain[i]);
                int pred = predVector.ToList().IndexOf(predVector.Max());
                int actual = yTrain[i].ToList().IndexOf(yTrain[i].Max());
                
                Console.WriteLine("Prediction: [" + predVector[0] + ", " + predVector[1] + "]");
                Console.WriteLine("Actual: [" + yTrain[i][0] + ", " + yTrain[i][1] + "]");
                Console.WriteLine();
                
                Assert.AreEqual(actual, pred);
            }
        }
    }
}