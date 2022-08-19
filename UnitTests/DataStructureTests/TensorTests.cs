using NUnit.Framework;
using NeuroSharp;
using NeuroSharp.Enumerations;
using NeuroSharp.Utilities;
using NeuroSharp.Datatypes;
using System.Collections.Generic;
using System.Linq;
using System;
using MathNet.Numerics.LinearAlgebra;


namespace UnitTests.DataStructureTests
{
    public class TensorTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void TensorIndexor_SetsCorrectValue_SquareTensors()
        {
            Tensor t1 = new Tensor(3);
            Tensor t2 = new Tensor(3, 3);
            Tensor t3 = new Tensor(3, 3, 3);

            t1[0] = 2;
            t1[2] = 4;

            t2[0, 0] = 5;
            t2[2, 0] = 2;
            t2[1, 1] = 8;
            t2[1, 2] = 1;
            t2[2, 2] = 6;

            t3[0, 0, 0] = 2;
            t3[1, 0, 0] = 1;
            t3[1, 1, 0] = 8;
            t3[0, 2, 0] = 3;
            t3[0, 0, 1] = 4;
            t3[2, 0, 1] = 5;
            t3[1, 0, 2] = 9;
            t3[2, 1, 2] = 7;
            t3[2, 2, 2] = 6;

            double[] expected1 = new double[] { 2, 0, 4 };

            double[] expected2 = new double[] { 5, 0, 2, 
                                                0, 8, 0, 
                                                0, 1, 6 };

            double[] expected3 = new double[] { 2, 1, 0, 
                                                0, 8, 0, 
                                                3, 0, 0, 

                                                4, 0, 5, 
                                                0, 0, 0, 
                                                0, 0, 0, 
                
                                                0, 9, 0, 
                                                0, 0, 7, 
                                                0, 0, 6 };

            Assert.AreEqual(expected1, t1.Values);
            Assert.AreEqual(expected2, t2.Values);
            Assert.AreEqual(expected3, t3.Values);
        }

        [Test]
        public void TensorIndexor_SetsCorrectValue_NonSquareTensors()
        {
            Tensor t1 = new Tensor(3);
            Tensor t2 = new Tensor(3, 5);
            Tensor t3 = new Tensor(3, 5, 2);

            t1[0] = 2;
            t1[2] = 4;

            t2[0, 0] = 5;
            t2[2, 0] = 2;
            t2[1, 1] = 9;
            t2[0, 2] = 8;
            t2[2, 2] = 4;
            t2[0, 3] = 3;
            t2[2, 3] = 1;
            t2[0, 4] = 6;
            t2[1, 4] = 7;

            t3[0, 0, 0] = 1;
            t3[2, 0, 0] = 8;
            t3[2, 2, 0] = 6;
            t3[0, 3, 0] = 7;
            t3[0, 4, 0] = 9;
            t3[2, 1, 1] = 4;
            t3[0, 2, 1] = 5;
            t3[1, 2, 1] = 2;
            t3[2, 4, 1] = 3;

            double[] expected1 = new double[] { 2, 0, 4 };

            double[] expected2 = new double[] { 5, 0, 2,
                                                0, 9, 0,
                                                8, 0, 4,
                                                3, 0, 1,
                                                6, 7, 0 };

            double[] expected3 = new double[] { 1, 0, 8,
                                                0, 0, 0,
                                                0, 0, 6,
                                                7, 0, 0,
                                                9, 0, 0,

                                                0, 0, 0,
                                                0, 0, 4,
                                                5, 2, 0,
                                                0, 0, 0,
                                                0, 0, 3 };

            Assert.AreEqual(expected1, t1.Values);
            Assert.AreEqual(expected2, t2.Values);
            Assert.AreEqual(expected3, t3.Values);
        }

        [Test]
        public void TensorIndexor_RetrievesCorrectValue()
        {
            Random rand = new Random();

            for (int n = 0; n < 200; n++)
            {
                int x = rand.Next(1, 500);

                Tensor t = new Tensor(x);

                int i = rand.Next(0, x);

                t[i] = 69;

                Assert.AreEqual(t[i], 69);
            }

            for (int n = 0; n < 200; n++)
            {
                int x = rand.Next(1, 500);
                int y = rand.Next(1, 500);

                Tensor t = new Tensor(x, y);

                int i = rand.Next(0, x);
                int j = rand.Next(0, y);

                t[i, j] = 69;

                Assert.AreEqual(t[i, j], 69);
            }

            for (int n = 0; n < 200; n++)
            {
                int x = rand.Next(1, 500);
                int y = rand.Next(1, 500);
                int z = rand.Next(1, 500);

                Tensor t = new Tensor(x, y, z);

                int i = rand.Next(0, x);
                int j = rand.Next(0, y);
                int k = rand.Next(0, z);

                t[i, j, k] = 69;

                Assert.AreEqual(t[i, j, k], 69);
            }
        }

        [Test]
        public void TensorMultiplication_ReturnsCorrectTensor_MultiplyingTwoTensors()
        {
            #region One Channel
            #region Matrix Product
            Tensor t1 = new Tensor(3, 2);
            Tensor t2 = new Tensor(2, 3);

            t1[0, 0] = 1;
            t1[1, 0] = 2;
            t1[2, 0] = 3;
            t1[0, 1] = 4;
            t1[1, 1] = 5;
            t1[2, 1] = 6;

            t2[0, 0] = 7;
            t2[1, 0] = 10;
            t2[0, 1] = 8;
            t2[1, 1] = 11;
            t2[0, 2] = 9;
            t2[1, 2] = 12;

            double[] expected = new double[]
            {
                50, 68,
                122, 167
            };

            Tensor t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #region Inner Vector Product
            t1 = new Tensor(3, 1);
            t2 = new Tensor(1, 3);

            t1[0, 0] = 1;
            t1[1, 0] = 2;
            t1[2, 0] = 3;

            t2[0, 0] = 4;
            t2[0, 1] = 5;
            t2[0, 2] = 6;

            expected = new double[]
            {
                32
            };

            t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #region Outer Vector Product
            t1 = new Tensor(1, 3);
            t2 = new Tensor(3, 1);

            t1[0, 0] = 4;
            t1[0, 1] = 5;
            t1[0, 2] = 6;

            t2[0, 0] = 1;
            t2[1, 0] = 2;
            t2[2, 0] = 3;

            expected = new double[]
            {
                4, 8, 12,
                5, 10, 15,
                6, 12, 18
            };

            t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #endregion
            #region Multiple Channels
            #region Matrix Product
            t1 = new Tensor(3, 2, 2);
            t2 = new Tensor(2, 3, 2);

            t1[0, 0, 0] = 1;
            t1[1, 0, 0] = 2;
            t1[2, 0, 0] = 3;
            t1[0, 1, 0] = 4;
            t1[1, 1, 0] = 5;
            t1[2, 1, 0] = 6;

            t1[0, 0, 1] = 11;
            t1[1, 0, 1] = 12;
            t1[2, 0, 1] = 13;
            t1[0, 1, 1] = 14;
            t1[1, 1, 1] = 15;
            t1[2, 1, 1] = 16;


            t2[0, 0, 0] = 7;
            t2[1, 0, 0] = 10;
            t2[0, 1, 0] = 8;
            t2[1, 1, 0] = 11;
            t2[0, 2, 0] = 9;
            t2[1, 2, 0] = 12;

            t2[0, 0, 1] = 17;
            t2[1, 0, 1] = 110;
            t2[0, 1, 1] = 18;
            t2[1, 1, 1] = 111;
            t2[0, 2, 1] = 19;
            t2[1, 2, 1] = 112;

            expected = new double[]
            {
                50, 68,
                122, 167,

                650, 3998,
                812, 4997
            };

            t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #region Inner Vector Product
            t1 = new Tensor(3, 1, 2);
            t2 = new Tensor(1, 3, 2);

            t1[0, 0, 0] = 1;
            t1[1, 0, 0] = 2;
            t1[2, 0, 0] = 3;

            t1[0, 0, 1] = 11;
            t1[1, 0, 1] = 12;
            t1[2, 0, 1] = 13;


            t2[0, 0, 0] = 4;
            t2[0, 1, 0] = 5;
            t2[0, 2, 0] = 6;

            t2[0, 0, 1] = 14;
            t2[0, 1, 1] = 15;
            t2[0, 2, 1] = 16;

            expected = new double[]
            {
                32,

                542
            };

            t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #region Outer Vector Product
            t1 = new Tensor(1, 3, 2);
            t2 = new Tensor(3, 1, 2);

            t1[0, 0, 0] = 4;
            t1[0, 1, 0] = 5;
            t1[0, 2, 0] = 6;

            t1[0, 0, 1] = 14;
            t1[0, 1, 1] = 15;
            t1[0, 2, 1] = 16;


            t2[0, 0, 0] = 1;
            t2[1, 0, 0] = 2;
            t2[2, 0, 0] = 3;

            t2[0, 0, 1] = 11;
            t2[1, 0, 1] = 12;
            t2[2, 0, 1] = 13;

            expected = new double[]
            {
                4, 8, 12,
                5, 10, 15,
                6, 12, 18,

                154, 168, 182,
                165, 180, 195,
                176, 192, 208
            };

            t3 = t1 * t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #endregion
        }

        [Test]
        public void TensorAddition_ReturnsCorrectTensor_AddingTwoTensors()
        {
            #region One Channel
            Tensor t1 = new Tensor(3, 2);
            Tensor t2 = new Tensor(3, 2);

            t1[0, 0] = 1;
            t1[1, 0] = 2;
            t1[2, 0] = 3;
            t1[0, 1] = 4;
            t1[1, 1] = 5;
            t1[2, 1] = 6;

            t2[0, 0] = 7;
            t2[1, 0] = 8;
            t2[2, 0] = 9;
            t2[0, 1] = 10;
            t2[1, 1] = 11;
            t2[2, 1] = 12;

            double[] expected = new double[]
            {
                8, 10, 12,
                14, 16, 18
            };

            Tensor t3 = t1 + t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
            #region Multiple Channels
            t1 = new Tensor(3, 2, 2);
            t2 = new Tensor(3, 2, 2);

            t1[0, 0, 0] = 1;
            t1[1, 0, 0] = 2;
            t1[2, 0, 0] = 3;
            t1[0, 1, 0] = 4;
            t1[1, 1, 0] = 5;
            t1[2, 1, 0] = 6;

            t1[0, 0, 1] = 11;
            t1[1, 0, 1] = 12;
            t1[2, 0, 1] = 13;
            t1[0, 1, 1] = 14;
            t1[1, 1, 1] = 15;
            t1[2, 1, 1] = 16;


            t2[0, 0, 0] = 7;
            t2[1, 0, 0] = 8;
            t2[2, 0, 0] = 9;
            t2[0, 1, 0] = 10;
            t2[1, 1, 0] = 11;
            t2[2, 1, 0] = 12;
             
            t2[0, 0, 1] = 17;
            t2[1, 0, 1] = 18;
            t2[2, 0, 1] = 19;
            t2[0, 1, 1] = 20;
            t2[1, 1, 1] = 21;
            t2[2, 1, 1] = 22;


            expected = new double[]
            {
                8, 10, 12,
                14, 16, 18,

                28, 30, 32,
                34, 36, 38
            };

            t3 = t1 + t2;

            Assert.AreEqual(expected, t3.Values);
            #endregion
        }

        [Test]
        public void Something()
        {
            Matrix<double> m1 = Matrix<double>.Build.Random(1, 1000000);
            Matrix<double> m2 = Matrix<double>.Build.Random(1000000, 1);
            Vector<double> m3 = Vector<double>.Build.Random(1000000);
            Vector<double> m4 = Vector<double>.Build.Random(1000000);

            var watch1 = new System.Diagnostics.Stopwatch();
            watch1.Start();
            Matrix<double> result1 = m1 * m2;
            watch1.Stop();

            var watch2 = new System.Diagnostics.Stopwatch();
            watch2.Start();
            double result2 = m3 * m4;
            watch2.Stop();

            var ms1 = watch1.ElapsedMilliseconds;
            var ms2 = watch2.ElapsedMilliseconds;
        }
    }
}
