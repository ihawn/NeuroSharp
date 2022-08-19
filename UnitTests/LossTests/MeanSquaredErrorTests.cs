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
    public class MeanSquaredErrorTests
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
                Vector<double> truth = Vector<double>.Build.Random(n);
                Vector<double> test = Vector<double>.Build.Random(n);

                double MSE(Vector<double> x)
                {
                    return LossFunctions.MeanSquaredError(truth, x);
                }
                Vector<double> nablaMSE(Vector<double> x)
                {
                    return LossFunctions.MeanSquaredErrorPrime(truth, x);
                }

                Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(MSE, test);
                Vector<double> explicitGradient = nablaMSE(test);

                Assert.IsTrue((finiteDiffGradient - explicitGradient).L2Norm() < 0.0001);
            }
        }
    }
}
