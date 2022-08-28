using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class RecurrentLayer : ParameterizedLayer
    {
        public Vector<double>[] Biases { get; set; }
        public Vector<double>[] BiasGradients { get; set; }
        public Vector<double>[] InputMemory { get; set; }
        public Vector<double>[] HiddenMemory { get; set; }
        public Vector<double>[] OutputMemory { get; set; }
        public Vector<double> PreviousHidden { get; set; }

        [JsonProperty]
        private Adam _adam;
        [JsonProperty] 
        private int _hiddenSize;
        [JsonProperty] 
        private int _vocabSize;
        [JsonProperty] 
        private int _sequenceLength;

        public RecurrentLayer(int hiddenSize, int vocabSize, int sequenceLength)
        {
            LayerType = LayerType.Recurrent;
            InputSize = vocabSize;
            OutputSize = vocabSize;
            _hiddenSize = hiddenSize;
            _vocabSize = vocabSize;
            _sequenceLength = sequenceLength;
        }
        

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            InputMemory = new Vector<double>[input.Count];
            HiddenMemory = new Vector<double>[input.Count];
            OutputMemory = new Vector<double>[input.Count];
            Input = input;

            for (int i = 0; i < input.Count; i++)
            {
                InputMemory[i] = Vector<double>.Build.Dense(_vocabSize);
                InputMemory[i][(int)Math.Round(input[i])] = 1;
                HiddenMemory[i] = ActivationFunctions.PointwiseTanh(Weights[(int)RNNWeight.U] * InputMemory[i] + 
                                                                 Weights[(int)RNNWeight.W] * 
                                                                    (i > 0 ? HiddenMemory[i - 1] : PreviousHidden));
                OutputMemory[i] = Weights[(int)RNNWeight.V] * HiddenMemory[i];
            }

            return MathUtils.Flatten(OutputMemory);
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            Vector<double>[] unflattenedError = MathUtils.UnflattenVecArray(outputError, Input.Count, _vocabSize);
            Vector<double> passedGradient = Vector<double>.Build.Dense(outputError.Count);

            for (int i = outputError.Count - 1; i >= 0; i--)
            {
                /*WeightGradients[(int)RNNWeight.V] += unflattenedError[i] * HiddenMemory[i];
                Vector<double> gradientWRTState = unflattenedError[i] * Weights[(int)RNNWeight.V] * 
                                                  ActivationFunctions.PointwiseTanhPrime(InputMemory[i]);*/
            }

            return passedGradient;
        }
        
        public override void InitializeParameters()
        {
            Weights = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_hiddenSize, _vocabSize), // U
                Matrix<double>.Build.Dense(_vocabSize, _hiddenSize), // V
                Matrix<double>.Build.Dense(_hiddenSize, _hiddenSize) // W
            };
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_hiddenSize, _vocabSize), // ∂U/L
                Matrix<double>.Build.Dense(_vocabSize, _hiddenSize), // ∂V/L
                Matrix<double>.Build.Dense(_hiddenSize, _hiddenSize) // ∂W/L
            };

            Biases = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_hiddenSize), // b
                Vector<double>.Build.Dense(_vocabSize)   // c
            };
            BiasGradients = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_hiddenSize), // ∂b/L
                Vector<double>.Build.Dense(_vocabSize) // ∂c/L
            };

            Bias = Vector<double>.Build.Random(OutputSize);
            BiasGradient = Vector<double>.Build.Dense(OutputSize);
            _adam = new Adam(InputSize, OutputSize);

            for (int n = 0; n < 3; n++)
            {
                for (int i = 0; i < Weights[n].RowCount; i++)
                {
                    for (int j = 0; j < Weights[n].ColumnCount; j++)
                    {
                        if(n == 0) // U
                            Weights[n][i, j] = MathUtils.GetInitialWeightFromRange(-Math.Sqrt(1d / _vocabSize),
                                Math.Sqrt(1d / _vocabSize));
                        else // V or W
                            Weights[n][i, j] = MathUtils.GetInitialWeightFromRange(-Math.Sqrt(1d / _hiddenSize),
                                Math.Sqrt(1d / _hiddenSize));
                    }
                }
            }
        }

        public override void DrainGradients()
        {

        }

        public override void SetGradientAccumulation(bool acc)
        {
            AccumulateGradients = acc;
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            switch (optimizerType)
            {
                case OptimizerType.GradientDescent:
                    Weights[0] -= learningRate * WeightGradients[0];
                    Bias -= learningRate * BiasGradient;
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate);
                    break;
            }

            if(AccumulateGradients)
                DrainGradients();
        }
    }
}
