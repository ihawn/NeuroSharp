using SimplexNoise;

public class Field
{
    public readonly List<DataPoint> DataPoints = new List<DataPoint>();
    public readonly List<DataPoint> SampleDataPoints = new List<DataPoint>();
    public double Width { get; private set; }
    public double Height { get; private set; }
    public float[,] SimplexNoise { get; set; }

    public Field(double width, double height)
    {
        Width = width;
        Height = height;
        Noise.Seed = new Random().Next();
        SimplexNoise = Noise.Calc2D((int)Math.Round(Width), (int)Math.Round(Height), 0.001f);
    }

    public void Resize(double width, double height) =>
        (Width, Height) = (width, height);
    
    private double RandomVelocity(Random rand, double min, double max)
    {
        double v = min + (max - min) * rand.NextDouble();
        if (rand.NextDouble() > .5)
            v *= -1;
        return v;
    }
    
    public void AddRandomDataPoints(int count, int height, int width, int radius, string color1, string color2)
    {
        Random rand = new Random();

        for (int i = 0; i < count; i++)
        {
            double x = Width * rand.NextDouble();
            double y = Width * rand.NextDouble();

            int xCoord = (int)Math.Round(x);
            int yCoord = (int)Math.Round(y);

            double perlin = SimplexNoise[xCoord, yCoord] / 255;
            string color = perlin > 0.5 ? color1 : color2;
            
            DataPoints.Add(
                    new DataPoint(
                        x: x,
                        y: y,
                        color: color
                    )
                );
        }
    }

    public void AddDataSamplers(int spacing)
    {
        for(int i = 0; i < Width; i += spacing)
        for (int j = 0; j < Height; j += spacing)
        {
            SampleDataPoints.Add(new DataPoint(x: i, y: j));
        }
    }
}
