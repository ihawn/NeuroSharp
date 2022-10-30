public class DataPoint
{
    public double X { get; private set; }
    public double Y { get; private set; }
    public string Color { get; set; }
    

    public DataPoint(double x, double y, string color)
    {
        (X, Y, Color) = (x, y, color);
    }

    public DataPoint(double x, double y)
    {
        (X, Y) = (x, y);
    }
}
