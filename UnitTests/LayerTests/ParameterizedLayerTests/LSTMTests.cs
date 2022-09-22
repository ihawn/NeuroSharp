using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;
using NeuroSharp.Datatypes;
using NeuroSharp.Training;

namespace UnitTests.LayerTests.ParameterizedLayerTests
{
    internal class LSTMTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void LSTM_ForwardPropagation_ReturnsCorrectResult()
        {
            Vector<double> x = Vector<double>.Build.Random(27 * 12);

            LongShortTermMemoryLayer layer = new LongShortTermMemoryLayer(100, 27, 256, 12);
            layer.InitializeParameters();

            Vector<double> output = layer.ForwardPropagation(x);
        }
    }
}
