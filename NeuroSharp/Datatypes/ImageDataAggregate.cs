using MathNet.Numerics.LinearAlgebra;

namespace NeuroSharp.Datatypes
{
    public struct ImageDataAggregate
    {
        public List<Vector<double>> XValues { get; set; }
        public List<Vector<double>> YValues { get; set; }

        public ImageDataAggregate(List<NetworkFormattedImage> data)
        {
            XValues = data.Select(x => x.GetFlattenedImage()).ToList();
            YValues = data.Select(y => y.Label).ToList();
        }
    }
}
