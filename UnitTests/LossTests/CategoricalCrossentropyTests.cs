using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace UnitTests
{
    public class CategoricalCrossentropyTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void CategoricalCrossentropyPrime_ReturnsCorrectGradient()
        {
            for(int n = 1; n < 100; n++)
            {
                Random rand = new Random();
                Vector<double> truth = Vector<double>.Build.Dense(n);
                truth[rand.Next(0, n)] = 1;

                double crossentropy(Vector<double> x)
                {
                    return LossFunctions.CategoricalCrossentropy(truth, x);
                }
                Vector<double> nablaCrossentropy(Vector<double> x)
                {
                    return LossFunctions.CategoricalCrossentropyPrime(truth, x);
                }

                Vector<double> test = Vector<double>.Build.Dense(n);
                for (int i = 0; i < n; i++)
                    test[i] = rand.NextDouble();

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(crossentropy, test);
                Vector<double> explicitGradient = nablaCrossentropy(test);

                Assert.IsTrue((finiteDiffGradient - explicitGradient).L2Norm() < 0.001, "Finite Difference: " + finiteDiffGradient + "\nActual: " + explicitGradient);
            }
        }
    }
}
