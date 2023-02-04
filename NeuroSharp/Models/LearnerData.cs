using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Models
{
    public struct LearnerData
    {
        public Vector<double> Data { get; set; }
        public Vector<double> Result { get; set; }
        public double Score { get; set; }

        public LearnerData(Vector<double> data, Vector<double> result, int score) { Data = data; Result = result; Score = score;  }
    }
}