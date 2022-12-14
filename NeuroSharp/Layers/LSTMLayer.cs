using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Optimizers;
using NeuroSharp.Enumerations;
using Newtonsoft.Json;

namespace NeuroSharp
{
    public class LSTMLayer : ParameterizedLayer
    {
        [JsonIgnore] public Vector<double>[][] ActivationCache { get; set; }
        [JsonIgnore] public Vector<double>[] HiddenStates { get; set; }
        [JsonIgnore] public Vector<double>[] CellStates { get; set; }
        [JsonIgnore] public Vector<double>[] CellInputs { get; set; }
        [JsonIgnore] public Vector<double>[] PreviousCellStates { get; set; }
        public FullyConnectedLayer[] LSTMGates { get; set; }
        [JsonIgnore]public ActivationLayer[][] ActivationGates { get; set; }
        public bool Bidirectional { get; set; }
        
        private Vector<double> _nextCellStateGradient;
        
        [JsonProperty] private int _hiddenUnits;
        [JsonProperty] private int _vocabSize;
        [JsonProperty] private int _sequenceLength;

        public LSTMLayer(int vocabSize, int hiddenUnits, int sequenceLength, bool bidirectional = false)
        {
            LayerType = LayerType.LSTM;
            _hiddenUnits = hiddenUnits;
            _vocabSize = vocabSize;
            _sequenceLength = sequenceLength;
            Bidirectional = bidirectional;
            ActivationGateInit();
        }
        
        //json constructor
        public LSTMLayer(int hiddenUnits, int vocabSize, int sequenceLength, int inputSize, int outputSize,
            FullyConnectedLayer[] lstmGates)
        {
            LayerType = LayerType.LSTM;
            InputSize = inputSize;
            OutputSize = outputSize;
            _hiddenUnits = hiddenUnits;
            _vocabSize = vocabSize;
            _sequenceLength = sequenceLength;
            LSTMGates = lstmGates;
            ActivationGateInit();
            CacheInit();
        }
        
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            /*return Vector<double>.Build.DenseOfEnumerable(
                Bidirectional
                ? LSTMPass(input).Concat(
                    LSTMPass(Vector<double>.Build.DenseOfEnumerable(input.Reverse()))
                ) : LSTMPass(input)
            );*/
            return LSTMPass(input);
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            if(!AccumulateGradients) DrainGradients();
           /* return Vector<double>.Build.DenseOfEnumerable(
                Bidirectional
                    ? LSTMBackPass(outputError.SubVector(outputError.Count / 2, outputError.Count / 2)).Concat(
                        LSTMBackPass(Vector<double>.Build.DenseOfEnumerable(
                            outputError.SubVector(0, outputError.Count / 2))
                        )
                    ) : LSTMBackPass(outputError)
            );*/
           return LSTMBackPass(outputError);
        }
        
        public override void InitializeParameters()
        {
            CacheInit();
            LSTMGates = new FullyConnectedLayer[]
            {
                new (inputSize: _vocabSize + _hiddenUnits, outputSize: _hiddenUnits),
                new (inputSize: _vocabSize + _hiddenUnits, outputSize: _hiddenUnits),
                new (inputSize: _vocabSize + _hiddenUnits, outputSize: _hiddenUnits),
                new (inputSize: _vocabSize + _hiddenUnits, outputSize: _hiddenUnits),
                new (inputSize: _hiddenUnits, outputSize: _vocabSize)
            };

            foreach (var layer in LSTMGates)
            {
                layer.InitializeParameters();
                layer.SetGradientAccumulation(true);
            }
        }
//todo: add support for inputs with varying sequence length
        public void CacheInit()
        {
            HiddenStates = new Vector<double>[_sequenceLength];
            CellStates = new Vector<double>[_sequenceLength];
            PreviousCellStates = new Vector<double>[_sequenceLength];
            ActivationCache = new Vector<double>[_sequenceLength][];
            CellInputs = new Vector<double>[_sequenceLength];
            for (int i = 0; i < _sequenceLength; i++)
                ActivationCache[i] = new Vector<double>[4];
        }

        public override void DrainGradients()
        {
            foreach(var denseLayer in LSTMGates)
                denseLayer.DrainGradients();
        }

        public override void SetSizeIO()
        {
            InputSize = _vocabSize * _sequenceLength;
            OutputSize = _vocabSize * (Bidirectional ? 2 : 1); //todo: add option for sequence to sequence lstm
        }

        public override void SetGradientAccumulation(bool acc)
        {
            AccumulateGradients = acc;
        }

        public override void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate)
        {
            foreach(var denseLayer in LSTMGates)
                denseLayer.UpdateParameters(optimizerType, sampleIndex, learningRate);

            if(AccumulateGradients)
                DrainGradients();
        }

        public Vector<double> LSTMPass(Vector<double> input)
        {
            for (int i = 0; i < _sequenceLength; i++)
            {
                Vector<double> currentInput = input.SubVector(i * _vocabSize, _vocabSize);
                LSTMForwardCell(currentInput, i);
            }

            Vector<double> outputCell = LSTMGates[(int)LSTMParameter.V].ForwardPropagation(HiddenStates[_sequenceLength - 1]);

            return outputCell;
        }

        public Vector<double> LSTMBackPass(Vector<double> outputError)
        {
            _nextCellStateGradient = Vector<double>.Build.Dense(_hiddenUnits);
            Vector<double> outputGradient = Vector<double>.Build.Dense(_sequenceLength * _vocabSize);

            outputError = LSTMGates[(int)LSTMParameter.V].BackPropagation(outputError);

            Vector<double> previousCellGradient = outputError;
            for (int i = _sequenceLength - 1; i >= 0; i--)
            {
                Vector<double> rawLSTMGradient = LSTMBackwardCell(previousCellGradient, i);
                outputGradient.SetSubVector(i * _vocabSize, _vocabSize, rawLSTMGradient);
                previousCellGradient = rawLSTMGradient.SubVector(rawLSTMGradient.Count - _hiddenUnits, _hiddenUnits);
            }

            return outputGradient;
        }

        public void LSTMForwardCell(Vector<double> currentCellInput, int index)
        {
            Vector<double> previousCellOutput = index > 0 ? HiddenStates[index - 1] : Vector<double>.Build.Dense(_hiddenUnits);
            
            List<double> rawData = currentCellInput.ToList();
            rawData.AddRange(previousCellOutput);
            
            // z_t
            Vector<double> concatData = Vector<double>.Build.DenseOfEnumerable(rawData);

            CellInputs[index] = concatData;

            // forget gate
            ActivationCache[index][(int)LSTMParameter.F] = ActivationGates[index][(int)LSTMParameter.F].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.F].ForwardPropagation(concatData));
            
            // input gate 1
            ActivationCache[index][(int)LSTMParameter.I] = ActivationGates[index][(int)LSTMParameter.I].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.I].ForwardPropagation(concatData));
            
            // input gate 2, c wave
            ActivationCache[index][(int)LSTMParameter.C] = ActivationGates[index][(int)LSTMParameter.C].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.C].ForwardPropagation(concatData));
            
            // output gate
            ActivationCache[index][(int)LSTMParameter.O] = ActivationGates[index][(int)LSTMParameter.O].ForwardPropagation(
                LSTMGates[(int)LSTMParameter.O].ForwardPropagation(concatData));

           PreviousCellStates[index] = index > 0 ? CellStates[index - 1] : Vector<double>.Build.Dense(_hiddenUnits) + 1;

           CellStates[index] =
               ActivationCache[index][(int)LSTMParameter.F].PointwiseMultiply(PreviousCellStates[index]) +
                    ActivationCache[index][(int)LSTMParameter.I].PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.C]);

           HiddenStates[index] =
                ActivationCache[index][(int)LSTMParameter.O]
                    .PointwiseMultiply(
                        ActivationGates[index][(int)LSTMParameter.V].ForwardPropagation(CellStates[index]));
        }

        public Vector<double> LSTMBackwardCell(Vector<double> previousError, int index)
        {
            Vector<double> hiddenStateGradient = previousError;

            Vector<double> cellStateGradient =
                    ActivationCache[index][(int)LSTMParameter.O]
                        .PointwiseMultiply(
                            ActivationGates[index][(int)LSTMParameter.V].BackPropagation(hiddenStateGradient))
                + _nextCellStateGradient;

            Vector<double> cGradient = cellStateGradient.PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.I]);
            Vector<double> iGradient = cellStateGradient.PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.C]);
            Vector<double> fGradient = cellStateGradient.PointwiseMultiply(PreviousCellStates[index]);
            Vector<double> oGradient = hiddenStateGradient.PointwiseMultiply(
                ActivationGates[index][(int)LSTMParameter.V].ForwardPropagation(CellStates[index]));

            for (int i = 0; i < 4; i++)
                LSTMGates[i].Input = CellInputs[index];

            Vector<double> F_ActivationGradient = LSTMGates[(int)LSTMParameter.F].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.F].BackPropagation(fGradient));
            
            Vector<double> I_ActivationGradient = LSTMGates[(int)LSTMParameter.I].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.I].BackPropagation(iGradient));

            Vector<double> O_ActivationGradient = LSTMGates[(int)LSTMParameter.O].BackPropagation(
               ActivationGates[index][(int)LSTMParameter.O].BackPropagation(oGradient));
            
            Vector<double> C_ActivationGradient = LSTMGates[(int)LSTMParameter.C].BackPropagation(
                ActivationGates[index][(int)LSTMParameter.C].BackPropagation(cGradient));

            _nextCellStateGradient = cellStateGradient.PointwiseMultiply(ActivationCache[index][(int)LSTMParameter.F]);
            
            return O_ActivationGradient + I_ActivationGradient + C_ActivationGradient + F_ActivationGradient;
        }

        void ActivationGateInit()
        {
            ActivationGates = new ActivationLayer[_sequenceLength][];
            for (int i = 0; i < _sequenceLength; i++)
                ActivationGates[i] = new ActivationLayer[]
                {
                    new (ActivationType.Sigmoid),
                    new (ActivationType.Sigmoid),
                    new (ActivationType.Tanh),
                    new (ActivationType.Sigmoid),
                    new (ActivationType.Tanh),
                };
        }
    }
}