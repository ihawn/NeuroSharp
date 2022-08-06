using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Drawing;

namespace NeuroSharp.Data
{
    public static class ImagePreprocessor
    {
        public static List<ImageData> LoadImages(string path)
        {
            List<ImageData> images = new List<ImageData>();

            List<string> imagePaths = Directory.GetFiles(path, "*.*", SearchOption.AllDirectories).ToList();
            foreach(string imagePath in imagePaths)
            {

            }

            return images;
        }
    }

    public struct ImageData
    {
        public Bitmap Image { get; set; }
        public Vector<double> Label { get; set; }
    }
}
