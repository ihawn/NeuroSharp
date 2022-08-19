using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;
using Newtonsoft.Json;

namespace UnitTests.ModelTests
{
    public class JSONModelTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SerializeToJSON_NetworkSerializesToJson_WithDenseLayers()
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
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(3, 1));
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            network.Train(xTrain, yTrain, epochs: 1000, optimizerType: OptimizerType.GradientDescent, learningRate: 0.1f);
            string modelJson = network.SerializeToJSON();
            Network deserializedNetwork = Network.DeserializeNetworkJSON(modelJson);

            foreach (var test in xTrain)
            {
                Vector<double> output1 = network.Predict(test);
                Vector<double> output2 = deserializedNetwork.Predict(test);

                Assert.AreEqual(output1, output2);
            }        
        }
    }
}
