using MathNet.Numerics.LinearAlgebra;
using System.IO;
using System.Drawing;
using System.Globalization;
using CsvHelper;
using NeuroSharp.Enumerations;
using NeuroSharp.Datatypes;
using NeuroSharp.Models;

namespace NeuroSharp.Data
{
    public static class ImagePreprocessor //todo: refactor
    {
        public static ImageDataAggregate GetImageData(string path, bool isColor = true, int? expectedWidth = null, 
            int? expectedHeight = null, int ? take = null)
        {
            string[] subDirectories = Directory.GetDirectories(path);

            List<NetworkFormattedImage[]> matrixOfImages = new List<NetworkFormattedImage[]>();

            for (int i = 0; i < subDirectories.Length; i++)
            {
                Console.WriteLine("Processing Image Data: " + (i+1) + "/" + subDirectories.Length);

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
                        categoryArray[j] = new NetworkFormattedImage(btmp, label, isColor);
                });

                matrixOfImages.Add(categoryArray.Where(x => x.Image != null).ToArray());
            }

            List<NetworkFormattedImage> images = new List<NetworkFormattedImage>();
            for(int i = 0; i < matrixOfImages.Count; i++)
                images = images.Concat(matrixOfImages[i]).ToList();

            Random rand = new Random();
            images = images.OrderBy(x => rand.Next()).Take(take != null ? take.Value : images.Count).ToList();
            return new ImageDataAggregate(images);
        }

        public static ImageDataAggregate GetImageData(string csvPath, string imagesRootPath, bool isColor = true, 
            int? expectedWidth = null, int? expectedHeight = null, int? take = null)
        {
            List<PathLabelModel> pathsAndLabels = new List<PathLabelModel>();
                    
            using (var reader = new StreamReader(csvPath))
            using (var csv = new CsvReader(reader, CultureInfo.InvariantCulture))
            {
                pathsAndLabels = csv.GetRecords<PathLabelModel>().ToList();
            }

            List<string> uniqeLabels = pathsAndLabels.Select(p => p.Label).Distinct().ToList();
                    
            NetworkFormattedImage[] images = new NetworkFormattedImage[pathsAndLabels.Count];

            int dataCount = pathsAndLabels.Count;
            if (take != null) dataCount = Math.Min(dataCount, take.Value);
            Parallel.For(0, pathsAndLabels.Count, i =>
            {
                Bitmap btmp = new Bitmap(imagesRootPath + "/" + pathsAndLabels[i].Path, true);
                bool heightCheck = true;
                bool widthCheck = true;
                if(expectedHeight != null && expectedHeight.Value != btmp.Height)
                    heightCheck = false;
                if(expectedWidth != null && expectedWidth.Value != btmp.Width)
                    widthCheck = false;

                if (heightCheck && widthCheck)
                {
                    Vector<double> label = Vector<double>.Build.Dense(uniqeLabels.Count);
                    label[uniqeLabels.IndexOf(pathsAndLabels[i].Label)] = 1;
                    images[i] = new NetworkFormattedImage(btmp, label, isColor);
                }
            });

            Random rand = new Random();
            var formatted = images.OrderBy(x => rand.Next())
                .Take(take != null ? take.Value : images.Length).ToList();
            return new ImageDataAggregate(formatted);
        }
    }
}
