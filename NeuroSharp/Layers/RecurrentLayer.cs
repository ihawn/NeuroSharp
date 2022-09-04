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
        public Vector<double>[] Biases { get; set; } //todo make this an inherited property (will need to update other layers)
        public Vector<double>[] BiasGradients { get; set; }
        public Vector<double>[] InputMemory { get; set; }
        public Vector<double>[] HiddenMemory { get; set; }
        public Vector<double>[] OutputMemory { get; set; }
        
        
        public Vector<double>[] StateInput { get; set; }
        public Vector<double>[] States { get; set; }
        public Vector<double>[] Outputs { get; set; }
        public Vector<double>[] UnflattenedInput { get; set; }
        public Vector<double>[] RecurrentGradient { get; set; }
        
        
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
            Input = input;
            Vector<double>[] unflattenedInput = UnflattenInputVector(input);
            UnflattenedInput = unflattenedInput;
            StateInput = new Vector<double>[_hiddenSize];
            States = new Vector<double>[_hiddenSize];
            Outputs = new Vector<double>[_hiddenSize];

            Vector<double> lastState = Vector<double>.Build.Dense(_sequenceLength);

            for (int i = 0; i < _hiddenSize; i++)
            {
                StateInput[i] = Weights[(int)RNNWeight.U] * unflattenedInput[i] + 
                                Weights[(int)RNNWeight.W] * (i == 0 ? lastState : States[i - 1]) +
                                Biases[(int)RNNBias.b];

                States[i] = ActivationFunctions.PointwiseTanh(StateInput[i]);

                Outputs[i] = Weights[(int)RNNWeight.V] * States[i] + Biases[(int)RNNBias.c];
            }

            /*Input = input;
            Vector<double>[] unflattenedOutput = new Vector<double>[_hiddenSize];
            Vector<double>[] unflattenedInput = UnflattenInputVector(input);
            HiddenMemory = new Vector<double>[unflattenedInput.Length];

            // n_y = _vocabSize
            // n_a = _hiddenSize

            Cache = new RNNCache { NextState = Vector<double>.Build.Dense(_hiddenSize) };

            for (int i = 0; i < _hiddenSize; i++)
            {
                Cache = RecurrentCellForward(unflattenedInput[i], Cache.NextState);
                HiddenMemory[i] = Cache.NextState; // a
                unflattenedOutput[i] = Cache.PredictionVector; // y_pred
            }

            Output = MathUtils.Flatten(unflattenedOutput);
            return Output;*/

            return MathUtils.Flatten(Outputs);
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            DrainGradients();
            Matrix<double> unflattenedError = MathUtils.VectorArrayToMatrix(UnflattenInputVector(outputError));
            Matrix<double> stateInput = MathUtils.VectorArrayToMatrix(StateInput);
            Matrix<double> unflattenedLayerInput = MathUtils.Unflatten(Input, _hiddenSize, _vocabSize);
            RecurrentGradient = new Vector<double>[_hiddenSize];

            Vector<double> nextStateGradient = Vector<double>.Build.Dense(_sequenceLength);

            for (int i = _hiddenSize - 1; i >= 0; i--)
            {
                WeightGradients[(int)RNNWeight.V] += 
                    MathUtils.TransposianShift(unflattenedError.Row(i).OuterProduct(States[i])).Transpose();
                
                BiasGradients[(int)RNNBias.c] += unflattenedError.Row(i);

                Vector<double> dh = Weights[(int)RNNWeight.V].Transpose() * unflattenedError.Row(i) + nextStateGradient;
                Vector<double> dhrec = dh.PointwiseMultiply(ActivationFunctions.PointwiseTanhPrime(stateInput.Row(i)));

                BiasGradients[(int)RNNBias.b] += dhrec;
                
                RecurrentGradient[i] = Weights[(int)RNNWeight.U].Transpose() * dhrec;
                WeightGradients[(int)RNNWeight.U] += unflattenedLayerInput.Row(i).OuterProduct(dh).Transpose(); //todo
                //[(int)RNNWeight.W] += 

                nextStateGradient = Weights[(int)RNNWeight.W].Transpose() * dhrec;
            }

            return MathUtils.Flatten(RecurrentGradient);
        }
        
        public override void InitializeParameters()
        {
            Weights = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_sequenceLength, _vocabSize), // U
                Matrix<double>.Build.Dense(_vocabSize, _sequenceLength), // V
                Matrix<double>.Build.Dense(_sequenceLength, _sequenceLength) // W
            };
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_sequenceLength, _vocabSize), // ∂U/L
                Matrix<double>.Build.Dense(_vocabSize, _sequenceLength), // ∂V/L
                Matrix<double>.Build.Dense(_sequenceLength, _sequenceLength) // ∂W/L
            };

            Biases = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_sequenceLength), // b
                Vector<double>.Build.Dense(_vocabSize)   // c
            };
            BiasGradients = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_sequenceLength), // ∂b/L
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
                            Weights[n][i, j] = MathUtils.GetInitialWeightFromRange(-Math.Sqrt(1d / _sequenceLength),
                                Math.Sqrt(1d / _sequenceLength));
                    }
                }
            }
        }

        public override void DrainGradients()
        {
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_sequenceLength, _vocabSize), // ∂U/L
                Matrix<double>.Build.Dense(_vocabSize, _sequenceLength), // ∂V/L
                Matrix<double>.Build.Dense(_sequenceLength, _sequenceLength) // ∂W/L
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

                case OptimizerType.Adam: //todo update this to include the RNN biases
                    _adam.Step(this, sampleIndex + 1, eta: learningRate);
                    break;
            }

            if(AccumulateGradients)
                DrainGradients();
        }
        
        //todo: write test for this or even better make this calculation implicit elsewhere
        public Vector<double>[] UnflattenInputVector(Vector<double> concatVec)
        {
            Vector<double>[] output = new Vector<double>[_hiddenSize];
            for (int i = 0; i < output.Length; i++)
                output[i] = concatVec.SubVector(i * _vocabSize, _vocabSize);
            return output;
        }


    }
}
