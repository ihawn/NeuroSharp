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
        public Vector<double> LSTMGradient { get; set; }
        public Vector<double>[] LSTMActivations { get; set; }
        public Vector<double>[] OutputErrorCache { get; set; }
        public Vector<double>[] ActivationErrorCache { get; set; }
        public LSTMCellModel[] LstmStateCache { get; set; }
        public Vector<double>[][] LstmErrorCache { get; set; }
        public Vector<double>[] EmbeddingErrorCache { get; set; }
        public Matrix<double>[][] WeightGradientCache { get; set; }
        public Matrix<double> Embeddings { get; set; }

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
            _vocabSize = outputUnits;
            _sequenceLength = sequenceLength;

            SigmoidGate = new ActivationLayer(ActivationType.Sigmoid);
            TanhGate = new ActivationLayer(ActivationType.Tanh);
        }
        
        public override Vector<double> ForwardPropagation(Vector<double> input)
        {
            Input = input;
            Vector<double> inputVec = input.SubVector(0, _vocabSize);
            LSTMCellModel initialState = new LSTMCellModel
            {
                Output = inputVec,
                EmbeddingTransformation = Embeddings * inputVec,
                LSTMActivations = new Vector<double>[4],
                ActivationVector = Vector<double>.Build.Dense(_hiddenUnits),
                CellVector = Vector<double>.Build.Dense(_hiddenUnits)
            };
            
            for (int i = 0; i < _sequenceLength - 1; i++)
            {
                LstmStateCache[i] = LSTMForwardCell(i == 0 ? initialState : LstmStateCache[i - 1]);
            }

            return Vector<double>.Build.DenseOfEnumerable(
                LstmStateCache.Select(y => y.Output).SelectMany(i => i));
        }

        public override Vector<double> BackPropagation(Vector<double> outputError)
        {
            if(!AccumulateGradients) DrainGradients();
            LSTMGradient = Vector<double>.Build.Dense(_sequenceLength * _vocabSize);

            //output cell error
            for (int i = _sequenceLength - 2; i >= 0; i--)
            {
                OutputErrorCache[i] = outputError.SubVector(i * _vocabSize, _vocabSize);
                ActivationErrorCache[i] = Weights[(int)LSTMWeight.H] * OutputErrorCache[i];
            }

            Vector<double> nextActivationError = Vector<double>.Build.Dense(ActivationErrorCache[0].Count);
            Vector<double> nextCellError = Vector<double>.Build.Dense(ActivationErrorCache[0].Count);

            
            for (int i = _sequenceLength - 1; i > 0; i--)
            {
                //core cell error
                Vector<double> activationError = ActivationErrorCache[i - 1] + nextActivationError;

                Vector<double> oa = LstmStateCache[i - 1].LSTMActivations[(int)LSTMWeight.O];
                Vector<double> eo =
                    activationError.PointwiseMultiply(TanhGate.ForwardPropagation(LstmStateCache[i - 1].CellVector));
                eo = eo.PointwiseMultiply(oa).PointwiseMultiply(1 - oa);

                Vector<double> cellError = activationError.PointwiseMultiply(oa);
                cellError = cellError.PointwiseMultiply(
                    TanhGate.BackPropagation(TanhGate.ForwardPropagation(LstmStateCache[i - 1].CellVector))
                );
                cellError += nextCellError;

                Vector<double> ia = LstmStateCache[i - 1].LSTMActivations[(int)LSTMWeight.I];
                Vector<double> ga = LstmStateCache[i - 1].LSTMActivations[(int)LSTMWeight.G];
                Vector<double> ei = cellError.PointwiseMultiply(ga);
                ei = ei.PointwiseMultiply(ia).PointwiseMultiply(1 - ia);

                Vector<double> eg = cellError.PointwiseMultiply(ia);
                eg = eg.PointwiseMultiply(TanhGate.BackPropagation(ga));

                Vector<double> fa = LstmStateCache[i - 1].LSTMActivations[(int)LSTMWeight.F];
                Vector<double> ef = cellError.PointwiseMultiply(LstmStateCache[i - 1].CellVector);
                ef = ef.PointwiseMultiply(fa).PointwiseMultiply(1 - fa);

                Vector<double> prevCellError = cellError.PointwiseMultiply(fa);

                Matrix<double> fgw = Weights[(int)LSTMWeight.F];
                Matrix<double> igw = Weights[(int)LSTMWeight.I];
                Matrix<double> ggw = Weights[(int)LSTMWeight.G];
                Matrix<double> ogw = Weights[(int)LSTMWeight.O];

                Vector<double> embededActivationError = fgw.Multiply(ef) +
                                                        igw.Multiply(ei) +
                                                        ogw.Multiply(eo) +
                                                        ggw.Multiply(eg);

                int hiddenInputUnits = fgw.RowCount;
                int hiddenUnits = fgw.ColumnCount;
                int inputUnits = hiddenInputUnits - hiddenUnits;

                Vector<double> prevActivationError =
                    embededActivationError.SubVector(inputUnits, embededActivationError.Count - inputUnits);
                
                Vector<double> embedError = embededActivationError.SubVector(0, inputUnits);
                //end core cell error

                LstmErrorCache[i - 1] = new Vector<double>[] { ef, ei, eo, eg };
                EmbeddingErrorCache[i - 1] = embedError;

                nextActivationError = prevActivationError;
                nextCellError = prevCellError;
            }
            
            //output cell weight gradient
            int batchSize = 1;
            for (int i = 0; i <= OutputErrorCache.Length - 1; i++)
            {
                Vector<double> error = OutputErrorCache[i];
                Vector<double> activation = LstmStateCache[i].ActivationVector;
                WeightGradients[(int)LSTMWeight.H] += activation.OuterProduct(error) / batchSize;
            }
            
            //gate weight gradients
            for (int i = 0; i < LstmErrorCache.Length; i++)
            {
                Vector<double> ef = LstmErrorCache[i][(int)LSTMWeight.F];
                Vector<double> ei = LstmErrorCache[i][(int)LSTMWeight.I];
                Vector<double> eo = LstmErrorCache[i][(int)LSTMWeight.O];
                Vector<double> eg = LstmErrorCache[i][(int)LSTMWeight.G];

                List<double> rawData = LstmStateCache[i].EmbeddingTransformation.ToList();
                rawData.AddRange(LstmStateCache[i].CellVector);
                Vector<double> concatData = Vector<double>.Build.DenseOfEnumerable(rawData);

                batchSize = 1;
                WeightGradientCache[i] = new Matrix<double>[]
                {
                    concatData.OuterProduct(ef) / batchSize,
                    concatData.OuterProduct(ei) / batchSize,
                    concatData.OuterProduct(eo) / batchSize,
                    concatData.OuterProduct(eg) / batchSize
                };
            }
            
            //gate weight sequence accumulation
            for (int i = 0; i < LstmErrorCache.Length; i++)
            {
                WeightGradients[(int)LSTMWeight.F] += WeightGradientCache[i][(int)LSTMWeight.F];
                WeightGradients[(int)LSTMWeight.I] += WeightGradientCache[i][(int)LSTMWeight.I];
                WeightGradients[(int)LSTMWeight.O] += WeightGradientCache[i][(int)LSTMWeight.O];
                WeightGradients[(int)LSTMWeight.G] += WeightGradientCache[i][(int)LSTMWeight.G];
            }

            return LSTMGradient;
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
            LstmStateCache = new LSTMCellModel[_sequenceLength - 1];
            OutputErrorCache = new Vector<double>[_sequenceLength - 1];
            ActivationErrorCache = new Vector<double>[_sequenceLength - 1];
            Embeddings = Matrix<double>.Build.Random(_inputUnits, _vocabSize);
            LstmErrorCache = new Vector<double>[_sequenceLength - 1][];
            EmbeddingErrorCache = new Vector<double>[_sequenceLength - 1];
            WeightGradientCache = new Matrix<double>[_sequenceLength - 1][];

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
            List<double> rawData = model.EmbeddingTransformation.ToList();
            rawData.AddRange(model.ActivationVector);
            Vector<double> concatData = Vector<double>.Build.DenseOfEnumerable(rawData);

            LSTMActivations[(int)LSTMWeight.F] = SigmoidGate.ForwardPropagation(concatData * Weights[(int)LSTMWeight.F]);
            LSTMActivations[(int)LSTMWeight.I] = SigmoidGate.ForwardPropagation(concatData * Weights[(int)LSTMWeight.I]);
            LSTMActivations[(int)LSTMWeight.O] = SigmoidGate.ForwardPropagation(concatData * Weights[(int)LSTMWeight.O]);
            LSTMActivations[(int)LSTMWeight.G] = TanhGate.ForwardPropagation(concatData * Weights[(int)LSTMWeight.G]);

            Vector<double> flattenedCellMemoryMatrix =
                model.CellVector.PointwiseMultiply(LSTMActivations[(int)LSTMWeight.F]) +
                LSTMActivations[(int)LSTMWeight.I].PointwiseMultiply(LSTMActivations[(int)LSTMWeight.G]);

            Vector<double> flattenedActivationMatrix =
                LSTMActivations[(int)LSTMWeight.O].PointwiseMultiply(TanhGate.ForwardPropagation(flattenedCellMemoryMatrix));

            Vector<double> output = OutputCell(flattenedActivationMatrix);
            
            return new LSTMCellModel
            {
                LSTMActivations = LSTMActivations,
                ActivationVector = flattenedActivationMatrix,
                CellVector = flattenedCellMemoryMatrix,
                Output = output,
                EmbeddingTransformation = Embeddings * output
            };
        }

        public Vector<double> OutputCell(Vector<double> activation)
        {
            return ActivationFunctions.Softmax(activation * Weights[(int)LSTMWeight.H]);
        }
    }
}
