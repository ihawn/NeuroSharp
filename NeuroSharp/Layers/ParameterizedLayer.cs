using Newtonsoft.Json;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Enumerations;

namespace NeuroSharp
{
    public abstract class ParameterizedLayer : Layer
    {
        public Matrix<double>[] Weights { get; set; }
        public Vector<double>[] Biases { get; set; }
        
        [JsonIgnore]
        public Matrix<double>[] WeightGradients { get; set; }
        [JsonIgnore]
        public Vector<double>[] BiasGradients { get; set; }
        [JsonIgnore]
        public bool AccumulateGradients { get; set; }

        public abstract void InitializeParameters();
        public abstract void UpdateParameters(OptimizerType optimizerType, int sampleIndex, double learningRate);
        public abstract void DrainGradients();
        public abstract void SetGradientAccumulation(bool acc);
    }
}
