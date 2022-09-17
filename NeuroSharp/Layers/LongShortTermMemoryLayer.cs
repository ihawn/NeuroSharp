using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Datatypes;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using NeuroSharp.Training;
using NeuroSharp.Utilities;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class LongShortTermMemoryLayer : ParameterizedLayer
    {
        public Vector<double>[] StateInput { get; set; }
        public Vector<double>[] States { get; set; }
        public Vector<double> Outputs { get; set; }
        public Vector<double> RecurrentGradient { get; set; }
        public Vector<double>[] LSTMActivations { get; set; }
        public LSTMCellModel[] LstmStateCache { get; set; }

        private ActivationLayer _stateActivation;
        [JsonProperty] private ActivationType _stateActivationType;
        [JsonProperty] private Adam _adam;
        [JsonProperty] private int _sequenceLength;
        [JsonProperty] private int _vocabSize;
        [JsonProperty] private int _hiddenSize;

        private ActivationLayer SigmoidGate;
        private ActivationLayer TanhGate;

        [JsonProperty] private int _inputUnits; // 100
        [JsonProperty] private int _hiddenUnits; // 256
        [JsonProperty] private int _outputUnits;  // vocab size

        public LongShortTermMemoryLayer(int inputUnits, int outputUnits, int hiddenUnits, int sequenceLength)
        {
            LayerType = LayerType.LSTM;
            _inputUnits = inputUnits;
            _hiddenUnits = hiddenUnits;
            _outputUnits = outputUnits;
            _sequenceLength = sequenceLength;

            SigmoidGate = new ActivationLayer(ActivationType.Sigmoid);
            TanhGate = new ActivationLayer(ActivationType.Tanh);
        }
        
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            LstmStateCache[0] = new LSTMCellModel
            {
                Input = input,
                LSTMActivations = new Vector<double>[4],
                FlattenedActivationMatrix = Vector<double>.Build.Dense(_inputUnits),
                FlattenedCellMatrix = Vector<double>.Build.Dense(_inputUnits)
            };
            
            for (int i = 0; i < _sequenceLength; i++)
            {
                LstmStateCache[i + 1] = LSTMForwardCell(LstmStateCache[i]);
                
            }

            return LstmStateCache.Last().Output;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            if(!AccumulateGradients) DrainGradients();
            
            Matrix<double> unflattenedError = MathUtils.Unflatten(outputError, _vocabSize, _sequenceLength).Transpose();
            Vector<double> nextStateGradient = Vector<double>.Build.Dense(_hiddenSize);

            RecurrentGradient = Vector<double>.Build.Dense(_sequenceLength * _vocabSize);

            for (int i = _sequenceLength - 1; i >= 0; i--)
            {
                Vector<double> stateGradient = Weights[(int)RNNWeight.V].TransposeThisAndMultiply(unflattenedError.Row(i))
                        .Add(nextStateGradient).PointwiseMultiply(_stateActivation.GradientPass(StateInput[i]));

                RecurrentGradient.SetSubVector(_vocabSize * i, _vocabSize, Weights[(int)RNNWeight.U].TransposeThisAndMultiply(stateGradient));
                
                BiasGradients[(int)RNNBias.b] += stateGradient;
                BiasGradients[(int)RNNBias.c] += unflattenedError.Row(i);

                WeightGradients[(int)RNNWeight.U] =
                    MathUtils.TransposianAdd(
                        WeightGradients[(int)RNNWeight.U],
                        stateGradient.OuterProduct(Input.SubVector(i * _vocabSize, _vocabSize)));
                
                WeightGradients[(int)RNNWeight.V] = 
                    MathUtils.TransposianAdd(
                        WeightGradients[(int)RNNWeight.V],
                        unflattenedError.Row(i).OuterProduct(States[i]));
                
                if(i > 0) WeightGradients[(int)RNNWeight.W] += stateGradient.OuterProduct(States[i - 1]).Transpose();

                nextStateGradient = Weights[(int)RNNWeight.W].TransposeThisAndMultiply(stateGradient);
            }

            return RecurrentGradient;
        }
        
        public override void InitializeParameters()
        {
            Weights = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Forget gate weight
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Input gate weight
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Output gate weight
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Gate gate weight
                Matrix<double>.Build.Dense(_hiddenUnits, _outputUnits), // Gate gate weight
            };
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Forget gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Input gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Output gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Gate gate weight gradient
                Matrix<double>.Build.Dense(_hiddenUnits, _outputUnits), // Gate gate weight gradient
            };
            LSTMActivations = new Vector<double>[4];

            _adam = new Adam(InputSize, OutputSize, weightCount: 5, biasCount: 0);
            LstmStateCache = new LSTMCellModel[_sequenceLength + 1];

            for (int n = 0; n < 3; n++)
            {
                for (int i = 0; i < Weights[n].RowCount; i++)
                {
                    for (int j = 0; j < Weights[n].ColumnCount; j++)
                    {
                        if(n < 4)
                            Weights[n][i, j] = MathUtils.GetInitialWeightFromRange(-Math.Sqrt(1d / _hiddenUnits),
                                Math.Sqrt(1d / _hiddenUnits));
                        else
                            Weights[n][i, j] = MathUtils.GetInitialWeightFromRange(-Math.Sqrt(1d / _outputUnits),
                                Math.Sqrt(1d / _outputUnits));
                    }
                }
            }
        }

        public override void DrainGradients()
        {
            WeightGradients = new Matrix<double>[]
            {
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Forget gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Input gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Output gate weight gradient
                Matrix<double>.Build.Dense(_inputUnits + _hiddenUnits, _hiddenUnits), // Gate gate weight gradient
                Matrix<double>.Build.Dense(_hiddenUnits, _outputUnits), // Gate gate weight gradient
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

        public LSTMCellModel LSTMForwardCell(LSTMCellModel model)
        {
            List<double> rawData = model.Input.ToList();
            rawData.AddRange(model.FlattenedActivationMatrix);
            Vector<double> concatData = Vector<double>.Build.DenseOfEnumerable(rawData);

            LSTMActivations[(int)LSTMWeight.F] = SigmoidGate.ForwardPropagation(Weights[(int)LSTMWeight.F] * concatData);
            LSTMActivations[(int)LSTMWeight.I] = SigmoidGate.ForwardPropagation(Weights[(int)LSTMWeight.I] * concatData);
            LSTMActivations[(int)LSTMWeight.O] = SigmoidGate.ForwardPropagation(Weights[(int)LSTMWeight.O] * concatData);
            LSTMActivations[(int)LSTMWeight.G] = TanhGate.ForwardPropagation(Weights[(int)LSTMWeight.G] * concatData);

            Vector<double> flattenedCellMemoryMatrix =
                model.FlattenedCellMatrix.PointwiseMultiply(LSTMActivations[(int)LSTMWeight.F]) +
                LSTMActivations[(int)LSTMWeight.I].PointwiseMultiply(LSTMActivations[(int)LSTMWeight.G]);

            Vector<double> flattenedActivationMatrix =
                LSTMActivations[(int)LSTMWeight.O].PointwiseMultiply(TanhGate.ForwardPropagation(flattenedCellMemoryMatrix));

            return new LSTMCellModel
            {
                LSTMActivations = LSTMActivations,
                FlattenedActivationMatrix = flattenedActivationMatrix,
                FlattenedCellMatrix = flattenedCellMemoryMatrix,
                Output = OutputCell(flattenedActivationMatrix)
            };
        }

        public Vector<double> OutputCell(Vector<double> input)
        {
            return ActivationFunctions.Softmax(
                    MathUtils.Flatten(
                    Weights[(int)LSTMWeight.H]
                        .PointwiseMultiply(MathUtils.Unflatten(input, _hiddenUnits, _outputUnits))
                )
            );
        }
    }
}
