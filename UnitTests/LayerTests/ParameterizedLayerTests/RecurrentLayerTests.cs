using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Training;

namespace UnitTests.LayerTests.ParameterizedLayerTests
{
    internal class RecurrentLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void ForwardPass_WorksCorrectly()
        {
            Vector<double> xTest = Vector<double>.Build.Random(25);

            Network network = new Network(25);
            network.Add(new RecurrentLayer(10, 25, 3));

            Vector<double> pred = network.Predict(xTest);
        }
    }
}
