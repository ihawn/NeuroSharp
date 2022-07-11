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
            //Mnist_Digits_Test(500, 500);
            Mnist_Digits_Test_Conv(5000, 500);
            //Conv_Vs_Non_Conv(20000, 5000, 25);

            #region testing
             /*float[,] filt = new float[,]
             {
                 {1, 2 },
                 {3, 4 }
             };
             Matrix<float> filter = Matrix<float>.Build.DenseOfArray(filt);

             float[,] mtxarr = new float[,]
             {
                 { 1, 2, 3, 9 },
                 { 5, 5, 6, 10 },
                 { 1, 2, 7, 2 },
                 { 8, 3, 0, 2 }
             };

            MaxPoolingLayer m = new MaxPoolingLayer(4, 2, 2);

            Matrix<float> mtx = Matrix<float>.Build.DenseOfArray(mtxarr);
            var o = m.MaxPool(mtx, 2, 2);*/
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

        static float Mnist_Digits_Test(int trainSize, int testSize)
        {
            //training data
            List<Vector<float>> xTrain = new List<Vector<float>>();
            List<Vector<float>> yTrain = new List<Vector<float>>();

            var trainData = MnistReader.ReadTrainingData().ToList();
            for(int n = 0; n < trainSize; n++)
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
            for (int n = 0; n < testSize; n++)
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
            network.Add(new FullyConnectedLayer(28*28, 250));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(250, 100));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(100, 10));
            //network.UseLoss(LossFunctions.MeanSquaredError, LossFunctions.MeanSquaredErrorPrime);
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            //train
            //network.MinibatchTrain(xTrain, yTrain, epochs: 5, OptimizerType.Adam, batchSize: 256);
            network.Train(xTrain, yTrain, epochs: 8, OptimizerType.Adam);

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
            float acc = (1f - ((float)wrongCount) / ((float)i));
            Console.WriteLine("Accuracy: " + acc);
            return acc;
        }

        static float Mnist_Digits_Test_Conv(int trainSize, int testSize)
        {
            //training data
            List<Vector<float>> xTrain = new List<Vector<float>>();
            List<Vector<float>> yTrain = new List<Vector<float>>();

            var trainData = MnistReader.ReadTrainingData().ToList();
            for (int n = 0; n < trainSize; n++)
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
            for (int n = 0; n < testSize; n++)
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
            network.Add(new ConvolutionalLayer(28*28, 26*26, 3, stride: 1));
            network.Add(new ActivationLayer(ActivationFunctions.Relu, ActivationFunctions.ReluPrime));
            network.Add(new MaxPoolingLayer(26*26, 2));
            network.Add(new FullyConnectedLayer(25*25, 150));
            network.Add(new ActivationLayer(ActivationFunctions.Tanh, ActivationFunctions.TanhPrime));
            network.Add(new FullyConnectedLayer(150, 10));
            network.Add(new SoftmaxActivationLayer());
            network.UseLoss(LossFunctions.CategoricalCrossentropy, LossFunctions.CategoricalCrossentropyPrime);

            //train
            network.Train(xTrain, yTrain, epochs: 8, OptimizerType.Adam);

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
            float acc = (1f - ((float)wrongCount) / ((float)i));
            Console.WriteLine("Accuracy: " + acc);
            return acc;
        }

        static void Conv_Vs_Non_Conv(int trainSize, int testSize, int testsToRun)
        {
            float denseNetAcc = 0;
            float convNetAcc = 0;

            for(int i = 0; i < testsToRun; i++)
            {
                denseNetAcc += Mnist_Digits_Test(trainSize, testSize);
                convNetAcc += Mnist_Digits_Test_Conv(trainSize, testSize);
            }

            denseNetAcc /= testsToRun;
            convNetAcc /= testsToRun;
            
            Console.WriteLine("Dense Network Average Accuracy: " + denseNetAcc);
            Console.WriteLine("Convolutional Network Average Accuracy: " + convNetAcc);
        }
    }
}
