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
        public void LSTM_BackPropagation_ReturnsCorrectInputGradient()
        {
            int vocabSize = 1;
            int sequenceLength = 1; //don't change till we get all the gradients correct for just a single cell
            int hiddenSize = 2;

            Vector<double> truthY = Vector<double>.Build.Random(2);//vocabSize);
            Vector<double> testX = Vector<double>.Build.Random(3);//sequenceLength * vocabSize);

            Network network = new Network(27 * 12);
            network.Add(new LongShortTermMemoryLayer(vocabSize, vocabSize, hiddenSize, sequenceLength));
            network.Add(new ActivationLayer(ActivationType.Tanh));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossType.CategoricalCrossentropy);

            double networkLoss(Vector<double> x)
            {
                x = network.Predict(x);
                return network.Loss(truthY, x);
            }

            Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLoss, testX);
            Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
            for (int k = network.Layers.Count - 1; k >= 0; k--)
            {
                testGradient = network.Layers[k].BackPropagation(testGradient);
            }

            Assert.IsTrue((finiteDiffGradient - testGradient).L2Norm() < 0.00001);
        } 

       /* [Test]
        public void LSTM_ForwardPropagation_ReturnsCorrectResult_NoBias()
        {
            Vector<double> x = Vector<double>.Build.DenseOfArray(
                new double[]
                {
                    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                }
            );

            Matrix<double> f = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    {  0.00369, -0.00459, -0.02139 },
                    {  0.00821, -0.00624, -0.00969 },
                    { -0.02060, -0.00321, -0.01258 },
                    { -0.00320, -0.00944, -0.00838 },
                    {  0.00184, -0.00986, -0.00441 }
                }
            );
            Matrix<double> i = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    {  0.00323,  0.00266,  0.00947 },
                    {  0.00788,  0.00640,  0.00870 },
                    { -0.00842,  0.00287,  0.01174 },
                    { -0.00170, -0.01302,  0.01620 },
                    {  0.00339, -0.00363, -0.00567 }
                }
            );
            Matrix<double> o = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { -0.00512,  0.01668,  0.00025 },
                    {  0.01304,  0.00636,  0.01062 },
                    {  0.01955, -0.00476, -0.00635 },
                    {  0.00303, -0.00738, -0.00394 },
                    {  0.01261, -0.00082, -0.00015 }
                }
            );
            Matrix<double> g = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { -0.00190, -0.01087, -0.00846 },
                    { -0.02060,  0.01022, -0.00349 },
                    { -0.01050,  0.00655, -0.01520},
                    {  0.00093, -0.00268,  0.00476 },
                    { -0.00396,  0.00781, -0.00913 }
                }
            );
            Matrix<double> h = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0.00518, -0.00018, 0.0088, 0.00516, -0.00762, 0.01436, -0.00183, -0.00083, 0.00858, 0.00957, 0.01487, 0.00088, -0.01623, 0.00059, -0.00202, 0.00954, 0.00983, 0.01052, 0.00800, -0.01281, -0.01707, 0.01194, 0.01298, -0.00067, -0.01300, 0.01241, -0.01878 },
                    { -0.01114, -0.00144, -0.00913, -0.00431, 0.00060, 0.01975, 0.00705, -0.00306, 0.00531, -0.00320, 0.00012, -0.00340, 0.00075, -0.01135, 0.00798, -0.00135, -0.00044, -0.01717, 0.00539, 0.00881, 0.00695, 0.00797, 0.01011, 0.01049, -0.00035, 0.00209, -0.0042 },
                    { 0.00231, -0.00798, -0.00101, -0.00618, 0.00347, -0.00104, -0.00575, 0.01600, -0.00687, -0.01311, -0.00595, 0.0265, -0.01021, 0.00039, -0.00013, 0.00695, 0.00410, 0.01390, 0.00802, -0.00980, -0.00326, -0.00178, 0.00595, -0.00684, -0.02036, -0.02114, 0.00759 }
                }
            );
            Matrix<double> embedding = Matrix<double>.Build.DenseOfArray(
                new double[,]
                {
                    { 0.00765, 0.00381 },
                    { 0.00438, -0.00469 },
                    { -0.01251, -0.01121 },
                    { 0.00122, 0.00164 },
                    {-0.01682, 0.00109 },
                    { 0.00041, -0.00474 },
                    { -0.00721, -0.00118 },
                    { 0.00631, -0.00018 },
                    { 0.00080, 0.00365 },
                    { -0.00865, -0.00426 },
                    { -0.01204, 0.00921 },
                    {-0.01185, -0.00608 },
                    { -0.00817, -0.01479 },
                    { -0.00804, 0.01047 },
                    { 0.01736, 0.00797 },
                    { 0.01287, -0.00545 },
                    {-0.00489, 0.00967 },
                    {-0.01099, 0.00245 },
                    {-0.01064, 0.00368 },
                    {-0.00782, -0.00211 },
                    {-0.00709, 0.01965 },
                    {-0.00265, -0.00560 },
                    { 0.00970, -0.00706 },
                    { 0.00174, -0.00793 },
                    {-0.00826, 0.00088 },
                    {-0.00581, -0.01148 },
                    {-0.00061, -0.00164 }
                }
            ).Transpose();

            LongShortTermMemoryLayer layer = new LongShortTermMemoryLayer(2, 27, 3, 12);
            layer.InitializeParameters();
            layer.Weights[(int)LSTMParameter.F] = f;
            layer.Weights[(int)LSTMParameter.I] = i;
            layer.Weights[(int)LSTMParameter.O] = o;
            layer.Weights[(int)LSTMParameter.G] = g;
            layer.Weights[(int)LSTMParameter.H] = h;
            layer.Embeddings = embedding;

            Vector<double> output = layer.ForwardPropagation(x);

            
            //embedding cache
            List<Vector<double>> expectedEmbeddingCache = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { 0.00080, 0.00365 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00817, -0.01479 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.01099, 0.00245 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.01251, -0.01121 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
                Vector<double>.Build.DenseOfArray(new double[] { -0.00581, -0.01148 }),
            };
            
            for (int j = 0; j < expectedEmbeddingCache.Count - 1; j++)
            {
                Assert.IsTrue((expectedEmbeddingCache[j] - layer.LstmStateCache[j].EmbeddingCache).L2Norm() < 0.00001);
            }
            
            
            //lstm cache
            List<Vector<double>[]> expectedLstmCache = new List<Vector<double>[]>
            {
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.50000824, 0.49999337, 0.49998686}),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.50000785, 0.50000638, 0.50000985 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.50001088, 0.50000916, 0.50000975 }),
                    Vector<double>.Build.DenseOfArray(new double[] { -7.68088289e-05, 2.86106319e-05, -1.95668273e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49996221, 0.50003246, 0.50007953 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49996428, 0.49997086, 0.49994848 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49996214, 0.49994246, 0.49996025 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 3.20447083e-04, -6.25577119e-05, 1.21147469e-04 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49999456, 0.50000867, 0.50005259 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49999583, 0.49999669, 0.49997943 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.50002249, 0.49995801, 0.50000573 }),
                    Vector<double>.Build.DenseOfArray(new double[] { -3.05533722e-05, 1.45275026e-04, 8.30246187e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49996532, 0.50003165, 0.50009382 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49996776, 0.49997363, 0.49994617 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997975, 0.49992995, 0.49996939 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.54248707e-04, 2.18035594e-05, 1.44389818e-04 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997069, 0.50002434, 0.50005854 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997255, 0.49997769, 0.4999615 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997058, 0.49995738, 0.49996902 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46547879e-04, -5.32489019e-05, 8.77406516e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997059, 0.50002439, 0.50005852 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997251, 0.49997779, 0.49996148 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997066, 0.49995739, 0.49996901 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46304040e-04, -5.30728572e-05, 8.73230070e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997054, 0.50002442, 0.50005851 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997249, 0.49997784, 0.49996147 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.4999707, 0.4999574, 0.499969 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46183215e-04, -5.29861657e-05,  8.71162700e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997051, 0.50002443, 0.50005851 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997247, 0.49997787, 0.49996147 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997072, 0.49995741, 0.499969 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46123348e-04, -5.29434819e-05, 8.70139407e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.4999705,  0.50002444, 0.50005851 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997247, 0.49997788, 0.49996146 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997073, 0.49995741, 0.499969 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46093685e-04, -5.29224677e-05, 8.69632905e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997049, 0.50002444, 0.50005851 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997247, 0.49997788, 0.49996146 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997074, 0.49995741, 0.49996899 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46078987e-04, -5.29121227e-05, 8.69382200e-05 }),
                },
                new Vector<double>[]
                {
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997049, 0.50002444, 0.50005851 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997247, 0.49997789, 0.49996146 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 0.49997074, 0.49995741, 0.49996899 }),
                    Vector<double>.Build.DenseOfArray(new double[] { 2.46071705e-04, -5.29070305e-05, 8.69258107e-05 }),
                },
            };

            for (int j = 0; j < expectedLstmCache.Count; j++)
            {
                for (int n = 0; n < 4; n++)
                {
                    Assert.IsTrue((expectedLstmCache[j][n] - layer.LstmStateCache[j].LSTMActivations[n]).L2Norm() < 0.00001);
                }
            }
            
            
            //activation cache
            List<Vector<double>> expectedActivationCache = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { -0.0000192, 0.0000072, -0.0000049 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0000705, -0.0000121, 0.0000278 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0000276, 0.0000303, 0.0000347 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0000774, 0.0000206, 0.0000534 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001003, -0.000003, 0.0000487 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001117, -0.0000148, 0.0000462 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001174, -0.0000206, 0.0000449 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001202, -0.0000236, 0.0000442 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001216, -0.0000250, 0.0000438 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001223, -0.0000257, 0.0000436 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001227, -0.0000261, 0.0000436 }),
            };

            for (int j = 0; j < expectedActivationCache.Count - 1; j++)
            {
                Assert.IsTrue((expectedActivationCache[j] - layer.LstmStateCache[j].ActivationVector).L2Norm() < 0.00001);
            }
            
            
            //cell cache
            List<Vector<double>> expectedCellCache = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[] { -0.0000384, 0.0000143, -0.0000098 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001410, -0.0000241, 0.0000557 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0000552, 0.0000606, 0.0000694 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0001547, 0.0000412, 0.0001069 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002006, -0.0000060, 0.0000973 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002235, -0.0000295, 0.0000923 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002348, -0.0000413, 0.0000897 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002405, -0.0000471, 0.0000884 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002433, -0.0000500, 0.0000877 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002447, -0.0000515, 0.0000873 }),
                Vector<double>.Build.DenseOfArray(new double[] { 0.0002453, -0.0000522, 0.0000871 })
            };

            for (int j = 0; j < layer.LstmStateCache.Length; j++)
            {
                Assert.IsTrue((expectedCellCache[j] - layer.LstmStateCache[j].CellVector).L2Norm() < 0.000001);
            }
            
            
            //output cache
            List<Vector<double>> expectedOutputCache = new List<Vector<double>>
            {
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703703, 0.03703704, 0.03703703, 0.03703703, 0.03703704, 0.03703703, 0.03703704, 0.03703703,
                    0.03703703, 0.03703703, 0.03703703, 0.03703703, 0.03703705, 0.03703703, 0.03703704, 0.03703703,
                    0.03703703, 0.03703702, 0.03703703, 0.03703705, 0.03703705, 0.03703703, 0.03703703, 0.03703704,
                    0.03703705, 0.03703703, 0.03703705
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703706, 0.03703703, 0.03703704, 0.03703704, 0.03703702, 0.03703706, 0.03703702, 0.03703705,
                    0.03703705, 0.03703705, 0.03703707, 0.03703706, 0.03703698, 0.03703704, 0.03703703, 0.03703707,
                    0.03703706, 0.03703708, 0.03703706, 0.03703699, 0.03703698, 0.03703706, 0.03703707, 0.03703702,
                    0.03703698, 0.03703704, 0.03703699
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703703, 0.03703702, 0.03703702, 0.03703703, 0.03703703, 0.03703707, 0.03703703, 0.03703705,
                    0.03703704, 0.03703702, 0.03703704, 0.03703707, 0.03703701, 0.03703702, 0.03703704, 0.03703705,
                    0.03703705, 0.03703704, 0.03703706, 0.03703702, 0.03703702, 0.03703705, 0.03703707, 0.03703704,
                    0.037037, 0.03703702, 0.03703703
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703704, 0.03703702, 0.03703703, 0.03703703, 0.03703702, 0.03703709, 0.03703702, 0.03703706,
                    0.03703705, 0.03703703, 0.03703706, 0.03703708, 0.03703697, 0.03703703, 0.03703703, 0.03703707,
                    0.03703707, 0.03703708, 0.03703708, 0.03703698, 0.03703698, 0.03703707, 0.03703709, 0.03703703,
                    0.03703696, 0.03703703, 0.03703699
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703706, 0.03703702, 0.03703704, 0.03703704, 0.03703701, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703705, 0.03703708, 0.03703708, 0.03703695, 0.03703704, 0.03703702, 0.03703708,
                    0.03703708, 0.0370371, 0.03703708, 0.03703697, 0.03703696, 0.03703707, 0.03703709, 0.03703702,
                    0.03703695, 0.03703704, 0.03703698
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703706, 0.03703702, 0.03703704, 0.03703705, 0.03703701, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703705, 0.03703708, 0.03703708, 0.03703695, 0.03703704, 0.03703702, 0.03703708,
                    0.03703708, 0.03703711, 0.03703708, 0.03703696, 0.03703695, 0.03703707, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703697
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703707, 0.03703702, 0.03703704, 0.03703705, 0.037037, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703705, 0.03703709, 0.03703708, 0.03703694, 0.03703704, 0.03703702, 0.03703709,
                    0.03703708, 0.03703711, 0.03703708, 0.03703695, 0.03703695, 0.03703708, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703696
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703707, 0.03703702, 0.03703704, 0.03703705, 0.037037, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703706, 0.03703709, 0.03703708, 0.03703694, 0.03703705, 0.03703702, 0.03703709,
                    0.03703708, 0.03703712, 0.03703708, 0.03703695, 0.03703694, 0.03703708, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703696
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703707, 0.03703702, 0.03703704, 0.03703705, 0.037037, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703706, 0.03703709, 0.03703708, 0.03703694, 0.03703705, 0.03703702, 0.03703709,
                    0.03703708, 0.03703712, 0.03703708, 0.03703695, 0.03703694, 0.03703708, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703696
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703707, 0.03703702, 0.03703704, 0.03703705, 0.037037, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703706, 0.03703709, 0.03703708, 0.03703694, 0.03703705, 0.03703702, 0.03703709,
                    0.03703708, 0.03703712, 0.03703708, 0.03703695, 0.03703694, 0.03703708, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703696
                }),
                Vector<double>.Build.DenseOfArray(new double[]
                {
                    0.03703707, 0.03703702, 0.03703704, 0.03703705, 0.037037, 0.03703708, 0.03703701, 0.03703706,
                    0.03703705, 0.03703706, 0.03703709, 0.03703708, 0.03703694, 0.03703705, 0.03703702, 0.03703709,
                    0.03703708, 0.03703712, 0.03703708, 0.03703695, 0.03703694, 0.03703708, 0.03703709, 0.03703701,
                    0.03703694, 0.03703705, 0.03703696
                }),
            };

            for (int j = 0; j < 11; j++)
            {
                Assert.IsTrue((expectedOutputCache[j] - output.SubVector(j * 27, 27)).L2Norm() < 0.000001);
            }

        }*/

       /*[Test]
       public void LSTM_BackPropagation_ReturnsCorrectHGradient()
       {
           Vector<double> testX = Vector<double>.Build.DenseOfArray(
               new double[]
               {
                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
               }
           );

           Matrix<double> f = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   {  0.00369, -0.00459, -0.02139 },
                   {  0.00821, -0.00624, -0.00969 },
                   { -0.02060, -0.00321, -0.01258 },
                   { -0.00320, -0.00944, -0.00838 },
                   {  0.00184, -0.00986, -0.00441 }
               }
           );
           Matrix<double> i = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   {  0.00323,  0.00266,  0.00947 },
                   {  0.00788,  0.00640,  0.00870 },
                   { -0.00842,  0.00287,  0.01174 },
                   { -0.00170, -0.01302,  0.01620 },
                   {  0.00339, -0.00363, -0.00567 }
               }
           );
           Matrix<double> o = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   { -0.00512,  0.01668,  0.00025 },
                   {  0.01304,  0.00636,  0.01062 },
                   {  0.01955, -0.00476, -0.00635 },
                   {  0.00303, -0.00738, -0.00394 },
                   {  0.01261, -0.00082, -0.00015 }
               }
           );
           Matrix<double> g = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   { -0.00190, -0.01087, -0.00846 },
                   { -0.02060,  0.01022, -0.00349 },
                   { -0.01050,  0.00655, -0.01520},
                   {  0.00093, -0.00268,  0.00476 },
                   { -0.00396,  0.00781, -0.00913 }
               }
           );
           Matrix<double> h = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   { 0.00518, -0.00018, 0.0088, 0.00516, -0.00762, 0.01436, -0.00183, -0.00083, 0.00858, 0.00957, 0.01487, 0.00088, -0.01623, 0.00059, -0.00202, 0.00954, 0.00983, 0.01052, 0.00800, -0.01281, -0.01707, 0.01194, 0.01298, -0.00067, -0.01300, 0.01241, -0.01878 },
                   { -0.01114, -0.00144, -0.00913, -0.00431, 0.00060, 0.01975, 0.00705, -0.00306, 0.00531, -0.00320, 0.00012, -0.00340, 0.00075, -0.01135, 0.00798, -0.00135, -0.00044, -0.01717, 0.00539, 0.00881, 0.00695, 0.00797, 0.01011, 0.01049, -0.00035, 0.00209, -0.0042 },
                   { 0.00231, -0.00798, -0.00101, -0.00618, 0.00347, -0.00104, -0.00575, 0.01600, -0.00687, -0.01311, -0.00595, 0.0265, -0.01021, 0.00039, -0.00013, 0.00695, 0.00410, 0.01390, 0.00802, -0.00980, -0.00326, -0.00178, 0.00595, -0.00684, -0.02036, -0.02114, 0.00759 }
               }
           );
           Matrix<double> embedding = Matrix<double>.Build.DenseOfArray(
               new double[,]
               {
                   { 0.00765, 0.00381 },
                   { 0.00438, -0.00469 },
                   { -0.01251, -0.01121 },
                   { 0.00122, 0.00164 },
                   {-0.01682, 0.00109 },
                   { 0.00041, -0.00474 },
                   { -0.00721, -0.00118 },
                   { 0.00631, -0.00018 },
                   { 0.00080, 0.00365 },
                   { -0.00865, -0.00426 },
                   { -0.01204, 0.00921 },
                   {-0.01185, -0.00608 },
                   { -0.00817, -0.01479 },
                   { -0.00804, 0.01047 },
                   { 0.01736, 0.00797 },
                   { 0.01287, -0.00545 },
                   {-0.00489, 0.00967 },
                   {-0.01099, 0.00245 },
                   {-0.01064, 0.00368 },
                   {-0.00782, -0.00211 },
                   {-0.00709, 0.01965 },
                   {-0.00265, -0.00560 },
                   { 0.00970, -0.00706 },
                   { 0.00174, -0.00793 },
                   {-0.00826, 0.00088 },
                   {-0.00581, -0.01148 },
                   {-0.00061, -0.00164 }
               }
           ).Transpose();
           
           
           
           int inputUnits = 1;
           int outputUnits = 2;
           int hiddenUnits = 1;
           int sequenceLength = 2;
           
           Vector<double> truthY = Vector<double>.Build.Random(outputUnits * (sequenceLength - 1));
           Vector<double> testX = Vector<double>.Build.Random(outputUnits * sequenceLength);
           Vector<double> testWeight = Vector<double>.Build.Random(hiddenUnits * outputUnits);

           Network network = new Network(outputUnits * sequenceLength);
           network.Add(new LongShortTermMemoryLayer(inputUnits, outputUnits, hiddenUnits, sequenceLength));
           network.Add(new ActivationLayer(ActivationType.Tanh));
           network.Add(new SoftmaxActivationLayer());
           network.UseLoss(LossType.CategoricalCrossentropy);
           
           LongShortTermMemoryLayer lstmLayer = (LongShortTermMemoryLayer)network.Layers[0];
           lstmLayer.Weights[(int)LSTMWeight.F] = f;
           lstmLayer.Weights[(int)LSTMWeight.I] = i;
           lstmLayer.Weights[(int)LSTMWeight.O] = o;
           lstmLayer.Weights[(int)LSTMWeight.G] = g;
           lstmLayer.Weights[(int)LSTMWeight.H] = h;
           lstmLayer.Embeddings = embedding;

           double networkLossHWeightInput(Vector<double> x)
           {
               LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
               lstm.Weights[(int)LSTMParameter.H] = MathUtils.Unflatten(x, hiddenUnits, outputUnits);
               x = network.Predict(testX);
               return network.Loss(truthY, x);
           }

           Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLossHWeightInput, testWeight);
           Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
           Vector<double> explicitWeightGradient = null;
           for (int k = network.Layers.Count - 1; k >= 0; k--)
           {
               testGradient = network.Layers[k].BackPropagation(testGradient);
               if (k == 0)
               {
                   LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
                   explicitWeightGradient = MathUtils.Flatten(lstm.WeightGradients[(int)LSTMParameter.H]);
               }
           }

           Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
       }
       
       [Test]
       public void LSTM_BackPropagation_ReturnsCorrectFGradient()
       {
           int inputUnits = 2;
           int outputUnits = 2;
           int hiddenUnits = 2;
           int sequenceLength = 3;
           
           Vector<double> truthY = Vector<double>.Build.Random(outputUnits * (sequenceLength - 1));
           Vector<double> testX = Vector<double>.Build.Random(outputUnits * sequenceLength);
           Vector<double> testWeight = Vector<double>.Build.Random((inputUnits + hiddenUnits) * hiddenUnits);

           Network network = new Network(outputUnits * sequenceLength);
           network.Add(new LongShortTermMemoryLayer(inputUnits, outputUnits, hiddenUnits, sequenceLength));
           network.Add(new ActivationLayer(ActivationType.Tanh));
           network.Add(new SoftmaxActivationLayer());
           network.UseLoss(LossType.CategoricalCrossentropy);
           
           double networkLossFWeightInput(Vector<double> x)
           {
               LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
               lstm.Weights[(int)LSTMParameter.F] = MathUtils.Unflatten(x, inputUnits + hiddenUnits, hiddenUnits);
               x = network.Predict(testX);
               return network.Loss(truthY, x);
           }

           Vector<double> finiteDiffGradient = MathUtils.FiniteDifferencesGradient(networkLossFWeightInput, testWeight);
           Vector<double> testGradient = LossFunctions.CategoricalCrossentropyPrime(truthY, network.Predict(testX));
           Vector<double> explicitWeightGradient = null;
           for (int k = network.Layers.Count - 1; k >= 0; k--)
           {
               testGradient = network.Layers[k].BackPropagation(testGradient);
               if (k == 0)
               {
                   LongShortTermMemoryLayer lstm = (LongShortTermMemoryLayer)network.Layers[0];
                   explicitWeightGradient = MathUtils.Flatten(lstm.WeightGradients[(int)LSTMParameter.F]);
               }
           }

           Assert.IsTrue((finiteDiffGradient - explicitWeightGradient).L2Norm() < 0.00001);
       }*/
    }
}

//todo train batching by stacking vector batches into a matrix