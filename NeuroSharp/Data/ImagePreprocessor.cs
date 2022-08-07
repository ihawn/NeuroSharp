using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Drawing;
using NeuroSharp.Enumerations;
using NeuroSharp.Datatypes;

namespace NeuroSharp.Data
{
    public static class ImagePreprocessor
    {
        public static List<NetworkFormattedImage> ReadImages(string path, ImagePreprocessingType imagePreprocessingType, int? expectedWidth = null, int? expectedHeight = null)
        {
            switch(imagePreprocessingType)
            {
                case ImagePreprocessingType.ParentFolderContainsLabel:

                    string[] subDirectories = Directory.GetDirectories(path);

                    List<NetworkFormattedImage[]> matrixOfImages = new List<NetworkFormattedImage[]>();

                    for (int i = 0; i < subDirectories.Length; i++)
                    {
                        Console.WriteLine("Loading Data... " + (i+1) + "/" + subDirectories.Length);

                        Vector<double> label = Vector<double>.Build.Dense(subDirectories.Length);
                        label[i] = 1;

                        string[] imagePaths = Directory.GetFiles(subDirectories[i], "*.*", SearchOption.AllDirectories);
                        NetworkFormattedImage[] categoryArray = new NetworkFormattedImage[imagePaths.Length];

                        Parallel.For(0, imagePaths.Length, j =>
                        {
                            Bitmap btmp = new Bitmap(imagePaths[j], true);

                            bool heightCheck = true;
                            bool widthCheck = true;
                            if(expectedHeight != null && expectedHeight.Value != btmp.Height)
                                heightCheck = false;
                            if(expectedWidth != null && expectedWidth.Value != btmp.Width)
                                widthCheck = false;

                            if(heightCheck && widthCheck)
                                categoryArray[j] = new NetworkFormattedImage(btmp, label);
                        });

                        matrixOfImages.Add(categoryArray.Where(x => x.Image != null).ToArray());
                    }

                    List<NetworkFormattedImage> images = new List<NetworkFormattedImage>();
                    for(int i = 0; i < matrixOfImages.Count; i++)
                        images = images.Concat(matrixOfImages[i]).ToList();

                    return images;
            }

            return new List<NetworkFormattedImage>();
        }

        public static ImageDataAggregate GetImageData(string path, ImagePreprocessingType imagePreprocessingType, int? expectedWidth = null, int? expectedHeight = null, int ? take = null)
        {
            Random rand = new Random();
            List<NetworkFormattedImage> data = ReadImages(path, imagePreprocessingType, expectedWidth, expectedHeight);

            data = data.OrderBy(x => rand.Next()).Take(take != null ? take.Value : data.Count).ToList();
            return new ImageDataAggregate(data);
        }
    }
}
