using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace UnitTests
{
    public class BinaryCrossentropyTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void BinaryCrossentropyPrime_ReturnsCorrectGradient()
        {
            for(int n = 1; n < 100; n++)
            {
                Random rand = new Random();
                Vector<double> truth = Vector<double>.Build.Dense(n);
                truth[rand.Next(0, n)] = 1;

                double BCE(Vector<double> x)
                {
                    return LossFunctions.CategoricalCrossentropy(truth, x);
                }
                Vector<double> nablaBCE(Vector<double> x)
                {
                    return LossFunctions.CategoricalCrossentropyPrime(truth, x);
                }

                Vector<double> test = Vector<double>.Build.Dense(n);
                for (int i = 0; i < n; i++)
                    test[i] = rand.NextDouble();

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(BCE, test);
                Vector<double> explicitGradient = nablaBCE(test);

                Assert.IsTrue((finiteDiffGradient - explicitGradient).L2Norm() < 0.0001);
            }
        }
    }
}
