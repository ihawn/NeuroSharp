using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp
{
    public class Network
    {
        public List<Layer> Layers { get; set; }
        public Func<Vector<float>, Vector<float>, float> Loss { get; set; }
        public Func<Vector<float>, Vector<float>, Vector<float>> LossPrime { get; set; }

        public Network()
        {
            Layers = new List<Layer>();
        }

        public void Add(Layer layer)
        {
            Layers.Add(layer);
        }

        public void UseLoss(Func<Vector<float>, Vector<float>, float> loss, Func<Vector<float>, Vector<float>, Vector<float>> lossPrime)
        {
            Loss = loss;
            LossPrime = lossPrime;
        }

        public List<Vector<float>> Predict(List<Vector<float>> inputData)
        {
            int samples = inputData.Count;
            List<Vector<float>> result = new List<Vector<float>>();

            for(int i = 0; i < samples; i++)
            {
                Vector<float> output = inputData[i];
                foreach (var layer in Layers)
                    output = layer.ForwardPropagation(output);
                result.Add(output);
            }

            return result;
        }

        public void Train(List<Vector<float>> xTrain, List<Vector<float>> yTrain, int epochs, float learningRate)
        {
            int samples = xTrain.Count;

            for(int i = 0; i < epochs; i++)
            {
                float err = 0;
                for(int j = 0; j < samples; j++)
                {
                    Vector<float> output = xTrain[j];
                    foreach (var layer in Layers)
                        output = layer.ForwardPropagation(output);

                    err += Loss(yTrain[j], output);

                    //back propagation
                    Vector<float> error = LossPrime(yTrain[j], output);
                    for(int k = Layers.Count - 1; k >= 0; k--)
                    {
                        error = Layers[k].BackPropagation(error);
                    }
                }

                err /= samples;
                Console.WriteLine("Epoch: " + i + "\nError: " + err + "\n");
            }
        }
    }
}
