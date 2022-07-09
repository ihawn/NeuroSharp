using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.MathUtils;
using NeuroSharp.Data;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public class NeuroSharp
    {
        static void Main(string[] args)
        {
            //XOR_Test();
            Mnist_Digits_Test();
            //Mnist_Digits_Test_Conv();

            #region testing
            /* float[,] filt = new float[,]
             {
                 {1, 2 },
                 {3, 4 }
             };
             Matrix<float> filter = Matrix<float>.Build.DenseOfArray(filt);

             float[] flat = new float[]
             {
                 1, 2, 3,
                 5, 5, 3,
                 1, 2, 7,
             };

             Vector<float> flat2 = Vector<float>.Build.DenseOfArray(flat);
             var o = ConvolutionalLayer.BackwardsConvolution(flat2, filter);*/
            #endregion
        }

        static void XOR_Test()
        {
            float[][] xx =
            {
                new float[]{ 0, 0 },
                new float[]{ 0, 1 },
                new float[]{ 1, 0 },
                new float[]{ 1, 1 }
            };
            List<Vector<float>> xTrain = new List<Vector<float>>();
            foreach (var x in xx)
                xTrain.Add(Vector<float>.Build.DenseOfArray(x));

            float[][] yy =
            {
                new float[]{ 0 },
                new float[]{ 1 },
                new float[]{ 1 },
                new float[]{ 0 }
            };
            List<Vector<float>> yTrain = new List<Vector<float>>();
            foreach (var y in yy)
                yTrain.Add(Vector<float>.Build.DenseOfArray(y));


            Network network = new Network();
            network.Add(new FullyConnectedLayer(2, 3));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            //network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(3, 1));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            //network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));

            //train
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
            network.Train(xTrain, yTrain, epochs: 1000, optimizerType: OptimizerType.Adam, learningRate: 0.1f);

            //test
            foreach(var test in xTrain)
            {
                var output = network.Predict(test);
                foreach (var o in output)
                    Console.WriteLine((float)o);
            }
        }

        static void Mnist_Digits_Test()
        {
            //training data
            List<Vector<float>> xTrain = new List<Vector<float>>();
            List<Vector<float>> yTrain = new List<Vector<float>>();

            var trainData = MnistReader.ReadTrainingData().ToList();
            for(int n = 0; n < 10000/*trainData.Count*/; n++)
            {
                var image = trainData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t/256f).ToArray();
                xTrain.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<float>.Build.DenseOfArray(categorical));
            }


            //testing data
            List<Vector<float>> xTest = new List<Vector<float>>();
            List<Vector<float>> yTest = new List<Vector<float>>();

            var testData = MnistReader.ReadTestData().ToList();
            for (int n = 0; n < 1024/*testData.Count*/; n++)
            {
                var image = testData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256f).ToArray();
                xTest.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTest.Add(Vector<float>.Build.DenseOfArray(categorical));
            }

           /* for(int n = 0; n < xTrain.Count; n++)
                xTrain[n] = PCA.GetPrincipleComponents(xTrain[n], 28);
            for (int n = 0; n < xTest.Count; n++)
                xTest[n] = PCA.GetPrincipleComponents(xTest[n], 28);*/

            //build network
            Network network = new Network();
            network.Add(new FullyConnectedLayer(28*28, 100));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(100, 50));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(50, 10));
            //network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            //train
            //network.MinibatchTrain(xTrain, yTrain, epochs: 5, OptimizerType.Adam, batchSize: 256);
            network.Train(xTrain, yTrain, epochs: 5, OptimizerType.Adam);

            //test
            int i = 0;
            int wrongCount = 0;
            foreach(var test in xTest)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if(prediction != actual)
                    wrongCount++;

                i++;
            }
            Console.WriteLine("Accuracy: " + (1f - ((float)wrongCount)/((float)i)));
        }

        static void Mnist_Digits_Test_Conv()
        {
            //training data
            List<Vector<float>> xTrain = new List<Vector<float>>();
            List<Vector<float>> yTrain = new List<Vector<float>>();

            var trainData = MnistReader.ReadTrainingData().ToList();
            for (int n = 0; n < 10000/*trainData.Count*/; n++)
            {
                var image = trainData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256f).ToArray();
                xTrain.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTrain.Add(Vector<float>.Build.DenseOfArray(categorical));
            }


            //testing data
            List<Vector<float>> xTest = new List<Vector<float>>();
            List<Vector<float>> yTest = new List<Vector<float>>();

            var testData = MnistReader.ReadTestData().ToList();
            for (int n = 0; n < 1024/*testData.Count*/; n++)
            {
                var image = testData[n];

                float[] flattenedNormalized = image.Data.Cast<byte>().Select(t => t / 256f).ToArray();
                xTest.Add(Vector<float>.Build.DenseOfArray(flattenedNormalized));

                float[] categorical = new float[10];
                categorical[image.Label] = 1;
                yTest.Add(Vector<float>.Build.DenseOfArray(categorical));
            }

            /* for(int n = 0; n < xTrain.Count; n++)
                 xTrain[n] = PCA.GetPrincipleComponents(xTrain[n], 28);
             for (int n = 0; n < xTest.Count; n++)
                 xTest[n] = PCA.GetPrincipleComponents(xTest[n], 28);*/

            //build network
            Network network = new Network();
            network.Add(new ConvolutionalLayer(28 * 28, 100, 2));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(729, 50));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(50, 10));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);

            //train
            network.Train(xTrain, yTrain, epochs: 5, OptimizerType.Adam);

            //test
            int i = 0;
            int wrongCount = 0;
            foreach (var test in xTest)
            {
                var output = network.Predict(test);
                int prediction = output.ToList().IndexOf(output.Max());
                int actual = yTest[i].ToList().IndexOf(yTest[i].Max());
                Console.WriteLine("Prediction: " + prediction);
                Console.WriteLine("Actual: " + actual + "\n");

                if (prediction != actual)
                    wrongCount++;

                i++;
            }
            Console.WriteLine("Accuracy: " + (1f - ((float)wrongCount) / ((float)i)));
        }
    }
}
