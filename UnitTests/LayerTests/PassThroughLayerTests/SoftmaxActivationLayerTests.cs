using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.MathUtils;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace UnitTests.LayerTests.PassThroughLayerTests
{
    public class SoftmaxActivationLayerTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void SoftMaxPrime_ReturnsCorrectJacobian()
        {
            Random rand = new Random();

            for(int n = 1; n < 100; n++)
            {
                Vector<double> test = Vector<double>.Build.Random(n);
                Matrix<double> explicitJacobian = ActivationFunctions.SoftmaxPrime(test);

                for (int i = 0; i < n; i++)
                {
                    double softmax(Vector<double> x)
                    {
                        return ActivationFunctions.Softmax(x)[i];
                    }
                    Vector<double> jacobianRow = explicitJacobian.Row(i);
                    Vector<double> jacobianRowFiniteDiff = Utils.FiniteDifferencesGradient(softmax, test);
                    Assert.IsTrue((jacobianRow - jacobianRowFiniteDiff).L2Norm() < 0.0001);
                }
            }
        }
    }
}
