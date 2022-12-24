using System;
using System.Collections.Generic;
using System.IO;


namespace NeuroSharp.Data
{
    //
    //from: https://stackoverflow.com/questions/49407772/reading-mnist-database
    //

    public static class MnistReader
    {
        private const string TrainImages_digits = "MnistDigits/train-images.idx3-ubyte";
        private const string TrainLabels_digits = "MnistDigits/train-labels.idx1-ubyte";
        private const string TestImages_digits = "MnistDigits/t10k-images.idx3-ubyte";
        private const string TestLabels_digits = "MnistDigits/t10k-labels.idx1-ubyte";

        private const string TrainImages_fashion = "MnistFashion/train-images-idx3-ubyte-fashion";
        private const string TrainLabels_fashion = "MnistFashion/train-labels-idx1-ubyte-fashion";
        private const string TestImages_fashion = "MnistFashion/t10k-images-idx3-ubyte-fashion";
        private const string TestLabels_fashion = "MnistFashion/t10k-labels-idx1-ubyte-fashion";

        public static IEnumerable<Image> ReadTrainingData(string data)
        {
            if(data == "digits")
                foreach (var item in Read(TrainImages_digits, TrainLabels_digits))
                {
                    yield return item;
                }
            else if(data == "fashion")
                foreach (var item in Read(TrainImages_fashion, TrainLabels_fashion))
                {
                    yield return item;
                }
        }

        public static IEnumerable<Image> ReadTestData(string data)
        {
            if (data == "digits")
                foreach (var item in Read(TestImages_digits, TestLabels_digits))
                {
                    yield return item;
                }
            else if (data == "fashion")
                foreach (var item in Read(TestImages_fashion, TestLabels_fashion))
                {
                    yield return item;
                }
        }

        private static IEnumerable<Image> Read(string imagesPath, string labelsPath)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
            BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

            int magicNumber = images.ReadBigInt32();
            int numberOfImages = images.ReadBigInt32();
            int width = images.ReadBigInt32();
            int height = images.ReadBigInt32();

            int magicLabel = labels.ReadBigInt32();
            int numberOfLabels = labels.ReadBigInt32();

            for (int i = 0; i < numberOfImages; i++)
            {
                var bytes = images.ReadBytes(width * height);
                var arr = new byte[height, width];

                arr.ForEach((j, k) => arr[j, k] = bytes[j * height + k]);

                yield return new Image()
                {
                    Data = arr,
                    Label = labels.ReadByte()
                };
            }
        }
    }

    public class Image
    {
        public byte Label { get; set; }
        public byte[,] Data { get; set; }
    }

    public static class Extensions
    {
        public static int ReadBigInt32(this BinaryReader br)
        {
            var bytes = br.ReadBytes(sizeof(Int32));
            if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }

        public static void ForEach<T>(this T[,] source, Action<int, int> action)
        {
            for (int w = 0; w < source.GetLength(0); w++)
            {
                for (int h = 0; h < source.GetLength(1); h++)
                {
                    action(w, h);
                }
            }
        }
    }
}
