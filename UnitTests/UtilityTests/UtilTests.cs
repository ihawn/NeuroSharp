using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using MathNet.Numerics.LinearAlgebra;
using System.Collections.Generic;
using System.Linq;
using System;

namespace UnitTests.UtilityTests
{
    public class UtilTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void FiniteDifferences_ReturnsCorrectDerivative()
        {
            for(int n = 1; n < 100; n++)
            {
                double xsquared(Vector<double> x)
                {
                    double sum = 0;
                    for (int i = 0; i < x.Count; i++)
                        sum += x[i] * x[i];
                    return sum;
                }

                double rastrigin(Vector<double> x)
                {
                    double sum = 0;
                    for (int i = 0; i < x.Count; i++)
                        sum += Math.Pow(x[i], 2) - 10 * Math.Cos(2 * Math.PI * x[i]);
                    return 20 + sum;
                }
                Vector<double> nabla_rastrigin(Vector<double> x)
                {
                    Vector<double> nablaX = Vector<double>.Build.Dense(x.Count);
                    for(int i = 0; i < x.Count; i++)
                        nablaX[i] = 2 * x[i] + 10 * Math.Sin(2 * Math.PI * x[i]) * 2 * Math.PI;
                    return nablaX;
                }

                Vector<double> x1 = Vector<double>.Build.Random(n);
                Vector<double> knownGradient1 = 2 * x1;
                Vector<double> testGradient1 = MathUtils.FiniteDifferencesGradient(xsquared, x1);

                Vector<double> x2 = Vector<double>.Build.Random(n);
                Vector<double> knownGradient2 = nabla_rastrigin(x2);
                Vector<double> testGradient2 = MathUtils.FiniteDifferencesGradient(rastrigin, x2);

                Assert.IsTrue((knownGradient1 - testGradient1).L2Norm() < 0.00001);
                Assert.IsTrue((knownGradient2 - testGradient2).L2Norm() < 0.00001);
            }
        }

        [Test]
        public void Unflatten_ReturnsCorrectMatrix_WhenPassedNonSquareFlattenedMatrix()
        {
            Vector<double> input1 = Vector<double>.Build.DenseOfArray(
                new double[]
                {
                    1, 2, 3, 4, 5, 6, 7, 8
                }
            );

            Matrix<double> expected1 = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 1, 3, 5, 7 },
                    { 2, 4, 6, 8 }
                }
            );
            
            Vector<double> input2 = Vector<double>.Build.DenseOfArray(
                new double[]
                {
                    1, 2, 3, 4, 5, 6, 7, 8
                }
            );

            Matrix<double> expected2 = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 1, 5 },
                    { 2, 6 },
                    { 3, 7 },
                    { 4, 8 }
                }
            );
            
            Assert.AreEqual(expected1, MathUtils.Unflatten(input1, 2, 4));
            Assert.AreEqual(expected2, MathUtils.Unflatten(input2, 4, 2));
        }
    }
}
