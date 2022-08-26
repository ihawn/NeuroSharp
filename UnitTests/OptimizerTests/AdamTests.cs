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

namespace UnitTests.OptimizerTests
{
    public class AdamTests
    {
        [SetUp]
        public void Setup()
        {
        }

        [Test]
        public void UpdateParameters_AdamCorrectlyUpdatesParameters_DenseNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new FullyConnectedLayer(32));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(16));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();

            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output);
                    network.BackPropagate(yTrain[j], output);
                    network.UpdateParameters(OptimizerType.Adam, j, learningRate: 0.0002);
                    err2 += network.Loss(yTrain[j], network.Predict(xTrain[j]));
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
        
        [Test]
        public void UpdateParameters_AdamCorrectlyUpdatesParameters_ConvDenseNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new MultiChannelConvolutionalLayer(5 * 5, kernel: 2, filters: 2, channels: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(24));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();

            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output);
                    network.BackPropagate(yTrain[j], output);
                    network.UpdateParameters(OptimizerType.Adam, j, learningRate: 0.0002);
                    err2 += network.Loss(yTrain[j], network.Predict(xTrain[j]));
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
        
        [Test]
        public void UpdateParameters_AdamCorrectlyUpdatesParameters_ConvNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new MultiChannelConvolutionalLayer(5 * 5, kernel: 2, filters: 8, channels: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(4 * 4 * 8, kernel: 2, filters: 1, channels: 8, stride: 2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();
            
            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output1 = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output1);
                    network.BackPropagate(yTrain[j], output1);
                    network.UpdateParameters(OptimizerType.Adam, j, learningRate: 0.0002);
                    Vector<double> output2 = network.Predict(xTrain[j]);
                    err2 += network.Loss(yTrain[j], output2);
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
        
        [Test]
        public void UpdateParameters_GDCorrectlyUpdatesParameters_DenseNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new FullyConnectedLayer(32));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(16));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();

            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output);
                    network.BackPropagate(yTrain[j], output);
                    network.UpdateParameters(OptimizerType.GradientDescent, j, learningRate: 0.0002);
                    err2 += network.Loss(yTrain[j], network.Predict(xTrain[j]));
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
        
        [Test]
        public void UpdateParameters_GDCorrectlyUpdatesParameters_ConvDenseNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new MultiChannelConvolutionalLayer(5 * 5, kernel: 2, filters: 2, channels: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(24));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new FullyConnectedLayer(4));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();

            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output);
                    network.BackPropagate(yTrain[j], output);
                    network.UpdateParameters(OptimizerType.GradientDescent, j, learningRate: 0.0002);
                    err2 += network.Loss(yTrain[j], network.Predict(xTrain[j]));
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
        
        [Test]
        public void UpdateParameters_GDCorrectlyUpdatesParameters_ConvNetwork()
        {
            double[] n1 = new double[]
            {
                0, 0, 1, 0, 0,
                0, 1, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 0, 1, 0, 0,
                0, 1, 1, 1, 0
            };
            double[] n2 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1,
                0, 1, 0, 0, 0,
                0, 1, 1, 1, 1
            };
            double[] n3 = new double[]
            {
                0, 1, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 0, 1, 1, 1,
                0, 0, 0, 0, 1,
                0, 1, 1, 1, 1
            };
            double[] n4 = new double[]
            {
                0, 1, 0, 1, 0,
                0, 1, 0, 1, 0,
                0, 1, 1, 1, 0,
                0, 0, 0, 1, 0,
                0, 0, 0, 1, 0
            };

            List<Vector<double>> xTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(n1),
                Vector<double>.Build.DenseOfArray(n2),
                Vector<double>.Build.DenseOfArray(n3),
                Vector<double>.Build.DenseOfArray(n4)
            };

            List<Vector<double>> yTrain = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 1, 0, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 1, 0, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 1, 0 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0, 0, 0, 1 })
            };

            Network network = new Network(25);
            network.Add(new MultiChannelConvolutionalLayer(5 * 5, kernel: 2, filters: 8, channels: 1, stride: 1));
            network.Add(new ActivationLayer(ActivationType.ReLu));
            network.Add(new MultiChannelConvolutionalLayer(4 * 4 * 8, kernel: 2, filters: 1, channels: 8, stride: 2));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);
            
            network.ParameterizedLayers = network.Layers.Where(l => l is ParameterizedLayer).Select(l => (ParameterizedLayer)l).ToList();
            
            for (int i = 0; i < 50; i++)
            {
                double err1 = 0;
                double err2 = 0;
                for (int j = 0; j < xTrain.Count; j++)
                {
                    Vector<double> output1 = network.Predict(xTrain[j]);
                    err1 += network.Loss(yTrain[j], output1);
                    network.BackPropagate(yTrain[j], output1);
                    network.UpdateParameters(OptimizerType.GradientDescent, j, learningRate: 0.0002);
                    Vector<double> output2 = network.Predict(xTrain[j]);
                    err2 += network.Loss(yTrain[j], output2);
                }

                err1 /= xTrain.Count;
                err2 /= yTrain.Count;
                
                Assert.IsTrue(err2 < err1);
            }
        }
    }
}
