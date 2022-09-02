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
        public RNNCache Cache { get; set; }
        public RNNGradientCache GradientCache { get; set; }

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
        
        //todo: write test for this (output shape, runtime issues, etc)
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Vector<double>[] unflattenedOutput = new Vector<double>[input.Count / _vocabSize];
            Vector<double>[] unflattenedInput = UnflattenInputVector(input);
            HiddenMemory = new Vector<double>[unflattenedInput.Length];

            // n_y = _vocabSize
            // n_a = _hiddenSize

            Cache = new RNNCache { NextState = Vector<double>.Build.Dense(_hiddenSize) };

            for (int i = 0; i < unflattenedInput.Length; i++)
            {
                Cache = RecurrentCellForward(unflattenedInput[i], Cache.NextState);
                HiddenMemory[i] = Cache.NextState; // a
                unflattenedOutput[i] = Cache.PredictionVector; // y_pred
            }

            Output = MathUtils.Flatten(unflattenedOutput);
            return Output;
        }
        
        // Wya = V
        // Wax = U

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            DrainGradients();
            Vector<double>[] unflattenedError = MathUtils.UnflattenVecArray(outputError, Input.Count, _vocabSize);
            Vector<double> prevState = Vector<double>.Build.Dense(_hiddenSize);

            for (int i = outputError.Count - 1; i >= 0; i--)
            {
                GradientCache = RecurrentCellBackwards(unflattenedError[i] + prevState, i);
                prevState = GradientCache.PrevStateGradient;
            }

            return GradientCache.InputGradient;
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
            PreviousHidden = Vector<double>.Build.Dense(_hiddenSize);
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
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_hiddenSize, _vocabSize), // ∂U/L
                Matrix<double>.Build.Dense(_vocabSize, _hiddenSize), // ∂V/L
                Matrix<double>.Build.Dense(_hiddenSize, _hiddenSize) // ∂W/L
            };
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
        
        // input.Count = _hiddenSize
        public RNNCache RecurrentCellForward(Vector<double> input, Vector<double> prevState)
        {
            Vector<double> nextState = ActivationFunctions.PointwiseTanh(Weights[(int)RNNWeight.U] * input + 
                                                                        Weights[(int)RNNWeight.W] * prevState +
                                                                        Biases[(int)RNNBias.b]);
            Vector<double> predictionVector =
                ActivationFunctions.Softmax(Weights[(int)RNNWeight.V] * nextState + Biases[(int)RNNBias.c]);

            return new RNNCache
            {
                NextState = nextState,
                PreviousState = prevState,
                Input = input,
                PredictionVector = predictionVector
            };
        }

        public RNNGradientCache RecurrentCellBackwards(Vector<double> nextState, int biasIndex)
        {
            Vector<double> activationGradient = ActivationFunctions.PointwiseTanhPrime(nextState).PointwiseMultiply(Cache.NextState);
            Vector<double> inputGradient = Weights[(int)RNNWeight.U] * activationGradient;
            WeightGradients[(int)RNNWeight.U] = activationGradient.OuterProduct(Input);
            WeightGradients[(int)RNNWeight.W] = activationGradient.OuterProduct(Cache.PreviousState);
            Vector<double> previousOutputGradient = WeightGradients[(int)RNNWeight.W] * activationGradient;
            BiasGradients[(int)RNNBias.b][biasIndex] = activationGradient.Sum();

            return new RNNGradientCache
            {
                PrevStateGradient = previousOutputGradient,
                InputGradient = inputGradient
            };
        }

        //todo: write test for this or even better make this calculation implicit elsewhere
        public Vector<double>[] UnflattenInputVector(Vector<double> concatVec)
        {
            Vector<double>[] output = new Vector<double>[concatVec.Count / _vocabSize];
            for (int i = 0; i < output.Length; i++)
                output[i] = concatVec.SubVector(i * _vocabSize, _vocabSize);
            return output;
        }
    }

    public struct RNNCache
    {
        public Vector<double> NextState { get; set; }
        public Vector<double> PreviousState { get; set; }
        public Vector<double> Input { get; set; }
        public Vector<double> PredictionVector { get; set; }
    }

    public struct RNNGradientCache
    {
        public Vector<double> InputGradient { get; set; }
        public Vector<double> PrevStateGradient { get; set; }
    }
}
