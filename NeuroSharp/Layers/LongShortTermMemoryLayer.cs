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
        public Vector<double>[][] ActivationCache { get; set; }
        public LSTMCellModel[] LstmStateCache { get; set; }
        public Vector<double>[] HiddenStates { get; set; }
        public Vector<double>[] CellStates { get; set; }
        public Vector<double>[] CellInputs { get; set; }
        public FullyConnectedLayer[] LSTMGates { get; set; }
        public Matrix<double> Embeddings { get; set; }

        private ActivationLayer _stateActivation;
        [JsonProperty] private ActivationType _stateActivationType;
        [JsonProperty] private Adam _adam;
        [JsonProperty] private int _sequenceLength;
        [JsonProperty] private int _vocabSize;
        [JsonProperty] private int _hiddenSize;


        private ActivationLayer[][] ActivationGates;

        private Vector<double> _nextCellStateGradient;
        private Vector<double> _nextHiddenStateGradient;

        [JsonProperty] private int _inputUnits; // 100
        [JsonProperty] private int _hiddenUnits; // 256
        [JsonProperty] private int _outputUnits;  // vocab size

        public LongShortTermMemoryLayer(int inputUnits, int outputUnits, int hiddenUnits, int sequenceLength)
        {
            LayerType = LayerType.LSTM;
            _inputUnits = inputUnits;
            _hiddenUnits = hiddenUnits;
            _outputUnits = outputUnits;
            _vocabSize = outputUnits;
            _sequenceLength = sequenceLength;

            ActivationGates = new ActivationLayer[_sequenceLength][];
            for (int i = 0; i < _sequenceLength; i++)
                ActivationGates[i] = new ActivationLayer[]
                {
                    new ActivationLayer(ActivationType.Sigmoid),
                    new ActivationLayer(ActivationType.Sigmoid),
                    new ActivationLayer(ActivationType.Tanh),
                    new ActivationLayer(ActivationType.Sigmoid),
                    new ActivationLayer(ActivationType.Tanh),
                };
        }
        
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;

            for (int i = 0; i < _sequenceLength; i++)
            {
                Vector<double> currentInput = input.SubVector(i * _vocabSize, _vocabSize);
                LSTMForwardCell(currentInput, i);
            }

            Vector<double> outputCell = HiddenStates[_sequenceLength - 1] * Weights[(int)LSTMParameter.V] +
                                        Biases[(int)LSTMParameter.V]; //todo: change this to just a dense layer

            return outputCell;
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            if(!AccumulateGradients) DrainGradients();
            _nextCellStateGradient = Vector<double>.Build.Dense(_hiddenUnits);
            _nextHiddenStateGradient = Vector<double>.Build.Dense(_hiddenUnits);
            Vector<double> outputGradient = Vector<double>.Build.Dense(_sequenceLength * _vocabSize);

            outputError = Weights[(int)LSTMParameter.V] * outputError;

            Vector<double> previousCellGradient = outputError;
            Vector<double> previousCellGradient2 = outputError;
            for (int i = _sequenceLength - 1; i >= 0; i--)
            {
                Vector<double> rawLSTMGradient = LSTMBackwardCell(previousCellGradient, i);
                outputGradient.SetSubVector(i * _vocabSize, _vocabSize, rawLSTMGradient);
                previousCellGradient = rawLSTMGradient.SubVector(0, _hiddenUnits);
                previousCellGradient2 = Weights[(int)LSTMParameter.V] * rawLSTMGradient.SubVector(0, _vocabSize);
            }

            return outputGradient;
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
            
            Biases = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_vocabSize),
            };
            BiasGradients = new Vector<double>[]
            {
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_hiddenUnits),
                Vector<double>.Build.Dense(_vocabSize),
            };

            HiddenStates = new Vector<double>[_sequenceLength];
            CellStates = new Vector<double>[_sequenceLength];
            ActivationCache = new Vector<double>[_sequenceLength][];
            for (int i = 0; i < _sequenceLength; i++)
                ActivationCache[i] = new Vector<double>[4];

            _adam = new Adam(InputSize, OutputSize, weightCount: 5, biasCount: 0);
            LstmStateCache = new LSTMCellModel[_sequenceLength - 1];
            Embeddings = Matrix<double>.Build.Random(_inputUnits, _vocabSize);

            for (int n = 0; n < 5; n++)
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

            LSTMGates = new FullyConnectedLayer[]
            {
                new FullyConnectedLayer(inputSize: _inputUnits + _hiddenUnits, outputSize: _hiddenUnits),
                new FullyConnectedLayer(inputSize: _inputUnits + _hiddenUnits, outputSize: _hiddenUnits),
                new FullyConnectedLayer(inputSize: _inputUnits + _hiddenUnits, outputSize: _hiddenUnits),
                new FullyConnectedLayer(inputSize: _inputUnits + _hiddenUnits, outputSize: _hiddenUnits)
            };
            
            foreach(var layer in LSTMGates)
                layer.InitializeParameters();

            CellInputs = new Vector<double>[_sequenceLength];
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

        public void LSTMForwardCell(Vector<double> currentCellInput, int index)
        {
            Vector<double> previousCellOutput = index > 0 ? HiddenStates[index - 1] : Vector<double>.Build.Dense(_hiddenUnits);
            Vector<double> previousCellState = index > 0 ? CellStates[index - 1] : Vector<double>.Build.Dense(_hiddenUnits);
            
            List<double> rawData = currentCellInput.ToList();
            rawData.AddRange(previousCellOutput);

            // z_t
            Vector<double> concatData = Vector<double>.Build.DenseOfEnumerable(rawData);

            CellInputs[index] = concatData;

            // forget gate
            ActivationCache[index][(int)LSTMParameter.F] = ActivationGates[index][(int)LSTMParameter.F].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.F].ForwardPropagation(concatData));
            
            // input gate 1
            ActivationCache[index][(int)LSTMParameter.I] = ActivationGates[index][(int)LSTMParameter.I] .ForwardPropagation(
                LSTMGates[(int)LSTMParameter.I].ForwardPropagation(concatData));
            
            // input gate 2, c wave
            ActivationCache[index][(int)LSTMParameter.C] = ActivationGates[index][(int)LSTMParameter.C].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.C].ForwardPropagation(concatData));
            
            // output gate
            ActivationCache[index][(int)LSTMParameter.O] = ActivationGates[index][(int)LSTMParameter.O].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.O].ForwardPropagation(concatData));

            CellStates[index] =
                ActivationCache[index][(int)LSTMParameter.F].PointwiseMultiply(previousCellState) +
                    ActivationCache[index][(int)LSTMParameter.I].PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.C]);

           HiddenStates[index] =
               ActivationCache[index][(int)LSTMParameter.O]
                   .PointwiseMultiply(
                        ActivationGates[index][(int)LSTMParameter.V].ForwardPropagation(CellStates[index]));
        }

        public Vector<double> LSTMBackwardCell(Vector<double> previousError, int index)
        {
            Vector<double> previousCellState = index > 0 ? CellStates[index - 1] : Vector<double>.Build.Dense(_hiddenUnits);

            Vector<double> hiddenStateGradient = previousError + _nextHiddenStateGradient;
            Vector<double> outputGateGradient = hiddenStateGradient.PointwiseMultiply(
                ActivationGates[index][(int)LSTMParameter.V].ForwardPropagation(CellStates[index]));

            Vector<double> cellStateGradient =
                    ActivationCache[index][(int)LSTMParameter.O]
                        .PointwiseMultiply(ActivationGates[index][(int)LSTMParameter.V].BackPropagation(hiddenStateGradient)) + 
                            _nextCellStateGradient;
            

            Vector<double> cGradient = cellStateGradient.PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.I]);
            Vector<double> iGradient = cellStateGradient.PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.C]);
            Vector<double> fGradient = cellStateGradient.PointwiseMultiply(previousCellState);

            for (int i = 0; i < 4; i++)
                LSTMGates[i].Input = CellInputs[index];
            
            Vector<double> F_ActivationGradient = LSTMGates[(int)LSTMParameter.F].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.F].BackPropagation(fGradient));
            
            Vector<double> I_ActivationGradient = LSTMGates[(int)LSTMParameter.I].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.I].BackPropagation(iGradient));
            
            Vector<double> O_ActivationGradient = LSTMGates[(int)LSTMParameter.O].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.O].BackPropagation(outputGateGradient));
            
            Vector<double> C_ActivationGradient = LSTMGates[(int)LSTMParameter.C].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.C].BackPropagation(cGradient));

            _nextCellStateGradient = cellStateGradient;
            _nextHiddenStateGradient = hiddenStateGradient;

            return F_ActivationGradient +
                   I_ActivationGradient +
                   O_ActivationGradient +
                   C_ActivationGradient;
        }
    }
}
