using SimplexNoise;

public class DataPoint
{
    public double X { get; private set; }
    public double Y { get; private set; }
    public double XVel { get; private set; }
    public double YVel { get; private set; }
    public double Radius { get; private set; }
    public string Color { get; private set; }
    

    public DataPoint(double x, double y, double xVel, double yVel, double radius, string color)
    {
        (X, Y, XVel, YVel, Radius, Color) = (x, y, xVel, yVel, radius, color);
    }

    public DataPoint(double x, double y)
    {
        (X, Y) = (x, y);
    }

    public void StepForward(double width, double height)
    {
        X += XVel;
        Y += YVel;
        if (X < 0 || X > width)
            XVel *= -1;
        if (Y < 0 || Y > height)
            YVel *= -1;

        if (X < 0)
            X += 0 - X;
        else if (X > width)
            X -= X - width;

        if (Y < 0)
            Y += 0 - Y;
        if (Y > height)
            Y -= Y - height;
    }
}

public class Field
{
    public readonly List<DataPoint> Balls = new List<DataPoint>();
    public readonly List<DataPoint> SampleBalls = new List<DataPoint>();
    public double Width { get; private set; }
    public double Height { get; private set; }
    public float[,] SimplexNoise { get; set; }

    public Field(double width, double height)
    {
        Width = width;
        Height = height;
        SimplexNoise = Noise.Calc2D((int)Math.Round(Width), (int)Math.Round(Height), 0.003f);
    }

    public void Resize(double width, double height) =>
        (Width, Height) = (width, height);

    public void StepForward()
    {
        foreach (DataPoint ball in Balls)
            ball.StepForward(Width, Height);
    }

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
            
            Balls.Add(
                    new DataPoint(
                        x: x,
                        y: y,
                        xVel: RandomVelocity(rand, 0, 0),
                        yVel: RandomVelocity(rand, 0, 0),
                        radius: radius,
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
            SampleBalls.Add(new DataPoint(x: i, y: j));
        }
    }
}

