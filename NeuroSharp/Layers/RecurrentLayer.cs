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
        public Vector<double>[] StateInput { get; set; }
        public Vector<double>[] States { get; set; }
        public Vector<double>[] Outputs { get; set; }
        public Vector<double>[] RecurrentGradient { get; set; }

        private ActivationLayer _stateActivation;
        [JsonProperty] private ActivationType _stateActivationType;
        [JsonProperty] private Adam _adam;
        [JsonProperty] private int _sequenceLength;
        [JsonProperty] private int _vocabSize;
        [JsonProperty] private int _hiddenSize;

        public RecurrentLayer(int sequenceLength, int vocabSize, int hiddenSize, ActivationType stateActivation = ActivationType.Tanh)
        {
            LayerType = LayerType.Recurrent;
            _stateActivation = new ActivationLayer(stateActivation);
            _stateActivationType = stateActivation;
            InputSize = vocabSize;
            OutputSize = vocabSize;
            _sequenceLength = sequenceLength;
            _vocabSize = vocabSize;
            _hiddenSize = hiddenSize;
        }
        
        //json constructor
        public RecurrentLayer(int sequenceLength, int vocabSize, int hiddenSize, ActivationType stateActivationType,
            Vector<double>[] stateInput, Vector<double>[] states, Vector<double>[] outputs,
            Vector<double>[] recurrentGradient, Matrix<double>[] weights, Vector<double>[] biases, Adam adam,
            int inputSize, int outputSize, bool accumulateGradients, int id)
        {
            _sequenceLength = sequenceLength;
            _vocabSize = vocabSize;
            _hiddenSize = hiddenSize;
            _stateActivation = new ActivationLayer(stateActivationType);
            _stateActivationType = stateActivationType;
            _adam = adam;

            Id = id;
            AccumulateGradients = accumulateGradients;
            InputSize = inputSize;
            OutputSize = outputSize;

            StateInput = stateInput;
            States = states;
            Outputs = outputs;
            RecurrentGradient = recurrentGradient;
            Weights = weights;
            Biases = biases;
        }

        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Vector<double>[] unflattenedInput = UnflattenInputVector(input);
            StateInput = new Vector<double>[_sequenceLength];
            States = new Vector<double>[_sequenceLength];
            Outputs = new Vector<double>[_sequenceLength];

            Vector<double> lastState = Vector<double>.Build.Dense(_hiddenSize);

            for (int i = 0; i < _sequenceLength; i++)
            {
                StateInput[i] = Weights[(int)RNNWeight.U] * unflattenedInput[i] + 
                                Weights[(int)RNNWeight.W] * (i == 0 ? lastState : States[i - 1]) +
                                Biases[(int)RNNBias.b];

                States[i] = _stateActivation.ForwardPropagation(StateInput[i]);

                Outputs[i] = Weights[(int)RNNWeight.V] * States[i] + Biases[(int)RNNBias.c];
            }

            return MathUtils.Flatten(Outputs);
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            if(!AccumulateGradients) DrainGradients();
            Matrix<double> unflattenedError = MathUtils.VectorArrayToMatrix(UnflattenInputVector(outputError));
            Matrix<double> stateInput = MathUtils.VectorArrayToMatrix(StateInput);
            Vector<double>[] unflattenedInput = UnflattenInputVector(Input);
            RecurrentGradient = new Vector<double>[_sequenceLength];

            Vector<double> nextStateGradient = Vector<double>.Build.Dense(_hiddenSize);

            for (int i = _sequenceLength - 1; i >= 0; i--)
            {
                WeightGradients[(int)RNNWeight.V] += 
                    MathUtils.TransposianShift(unflattenedError.Row(i).OuterProduct(States[i])).Transpose();
                
                Vector<double> dh = Weights[(int)RNNWeight.V].Transpose() * unflattenedError.Row(i) + nextStateGradient;
                Vector<double> dhrec = dh.PointwiseMultiply(ActivationFunctions.PointwiseTanhPrime(stateInput.Row(i)));

                BiasGradients[(int)RNNBias.b] += dhrec;
                BiasGradients[(int)RNNBias.c] += unflattenedError.Row(i);
                
                RecurrentGradient[i] = Weights[(int)RNNWeight.U].Transpose() * dhrec;
                WeightGradients[(int)RNNWeight.U] += 
                    MathUtils.TransposianShift(dhrec.OuterProduct(unflattenedInput[i])).Transpose();
                if(i > 0) WeightGradients[(int)RNNWeight.W] += dhrec.OuterProduct(States[i - 1]).Transpose();

                nextStateGradient = Weights[(int)RNNWeight.W].Transpose() * dhrec;
            }

            return MathUtils.Flatten(RecurrentGradient);
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
                Matrix<double>.Build.Dense(_vocabSize, _hiddenSize), // ∂V/V
                Matrix<double>.Build.Dense(_hiddenSize, _hiddenSize) // ∂W/W
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
            
            _adam = new Adam(InputSize, OutputSize, weightCount: 3, biasCount: 2);

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
                Matrix<double>.Build.Dense(_vocabSize, _hiddenSize), // ∂V/V
                Matrix<double>.Build.Dense(_hiddenSize, _hiddenSize) // ∂W/W
            };
            BiasGradients = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_hiddenSize), // ∂b/L
                Vector<double>.Build.Dense(_vocabSize) // ∂c/L
            };
        }

        public override void SetSizeIO()
        {
            InputSize = _vocabSize * _sequenceLength;
            OutputSize = InputSize;
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
                    for(int i = 0; i < Weights.Length; i++)
                        Weights[i] -= learningRate * WeightGradients[i];
                    for(int i = 0; i < Biases.Length; i++)
                        Biases[i] -= learningRate * base.BiasGradients[i];
                    break;

                case OptimizerType.Adam:
                    _adam.Step(this, sampleIndex + 1, eta: learningRate);
                    break;
            }

            if(AccumulateGradients)
                DrainGradients();
        }
        
        //todo: write test for this or even better make this calculation implicit elsewhere
        public Vector<double>[] UnflattenInputVector(Vector<double> concatVec)
        {
            Vector<double>[] output = new Vector<double>[_sequenceLength];
            for (int i = 0; i < output.Length; i++)
                output[i] = concatVec.SubVector(i * _vocabSize, _vocabSize);
            return output;
        }
    }
}
