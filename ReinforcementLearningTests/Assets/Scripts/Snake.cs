using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;
using System.Linq;

public class Snake
{
    public SnakeGame SnakeGame { get; private set; }
    public List<SnakeGameSquare> Segments { get; set; }
    public int Length { get { return Segments.Count; } }
    public int VisionDistance { get; private set; }

    private Coord lastTailCoord;

    public Snake(int startingLength, SnakeGame parentGame, int visionDistance = 3)
    {
        if (startingLength < 2)
            throw new System.Exception("Snake must start with at least 2 segments");

        SnakeGame = parentGame;

        Segments = new List<SnakeGameSquare>();
        for (int x = startingLength - 1; x >= 0; x--)
            Segments.Add(new SnakeGameSquare(x, parentGame.YSize / 2, parentGame.SquareObject, Color.red));
        VisionDistance = visionDistance;
    }

    public SnakeMoveResult MoveSnake(Move move)
    {
        lastTailCoord = Segments.Last().Position;

        float xDiff = Segments[0].Position.X - Segments[1].Position.X;
        float yDiff = Segments[0].Position.Y - Segments[1].Position.Y;

        Matrix<float> transformation = Matrix<float>.Build.DenseOfArray(
        new float[,]
        {
            { yDiff, xDiff, -yDiff },
            { -xDiff, yDiff, xDiff },
        });

        Vector<float> oneHotMove = Vector<float>.Build.Dense(3);
        oneHotMove[(int)move] = 1;

        Vector<float> moveOffset = transformation * oneHotMove;

        Coord newHeadPos = new Coord
        {
            X = (int)Mathf.Round(Segments[0].Position.X + moveOffset[0]),
            Y = (int)Mathf.Round(Segments[0].Position.Y + moveOffset[1]),
        };

        if (newHeadPos.X >= SnakeGame.XSize || newHeadPos.X < 0 ||
            newHeadPos.Y >= SnakeGame.YSize || newHeadPos.Y < 0)
        {
            return new SnakeMoveResult { OutOfBounds = true };
        }

        if(SnakeGame.GameSpaceMatrix.Cast<SnakeGameSquare>().Any(square =>
                square.Position.X == newHeadPos.X &&
                square.Position.Y == newHeadPos.Y &&
                square.ContainsSnakeSegment))
        {
            return new SnakeMoveResult { HitSelf = true };
        }

        for (int i = Segments.Count - 1; i > 0; i--)
        {
            Segments[i].Position.X = Segments[i - 1].Position.X;
            Segments[i].Position.Y = Segments[i - 1].Position.Y;
        }

        Segments[0].Position = newHeadPos;

        RenderSnake();
        return new SnakeMoveResult { AteFood = CheckAteFood() };
    }

    public Vector<double> GetSnakeVision()
    {
        float xDiff = Segments[0].Position.X - Segments[1].Position.X;
        float yDiff = Segments[0].Position.Y - Segments[1].Position.Y;

        Coord foodPosition = SnakeGame.GameSpaceMatrix.Cast<SnakeGameSquare>().First(x => x.ContainsFood).Position;
        float foodXDiff = Segments[0].Position.X - foodPosition.X;
        float foodYDiff = Segments[0].Position.Y - foodPosition.Y;
        foodXDiff /= Mathf.Abs(foodXDiff);
        foodYDiff /= Mathf.Abs(foodYDiff);
        foodXDiff = Mathf.Round(foodXDiff);
        foodYDiff = Mathf.Round(foodYDiff);

        Matrix<float> transformation = Matrix<float>.Build.DenseOfArray(
        new float[,]
        {
                { yDiff, xDiff, -yDiff },
                { -xDiff, yDiff, xDiff },
        });

        // 0 to 2 = obstacles left, straight, right
        // 3 to 6 = orientation
        // 7 to 10 = absolute food proximity
        // 11 to 13 = food directly left, up, right
        Vector<double> vision = Vector<double>.Build.Dense(14); 

        List<Vector<float>> offsets = new List<Vector<float>>
        {
            transformation * Vector<float>.Build.DenseOfArray(new float[] { 1, 0, 0 } ),
            transformation * Vector<float>.Build.DenseOfArray(new float[] { 0, 1, 0 } ),
            transformation * Vector<float>.Build.DenseOfArray(new float[] { 0, 0, 1 } )
        };

        for (int i = 0; i < 3; i++)
        {
            Coord look = new Coord
            {
                X = (int)Mathf.Round(Segments[0].Position.X + offsets[i][0]),
                Y = (int)Mathf.Round(Segments[0].Position.Y + offsets[i][1]),
            };

            //obstacle check
            vision[i] =
                look.X < 0 || look.X >= SnakeGame.XSize ||
                look.Y < 0 || look.Y >= SnakeGame.YSize ||
                SnakeGame.GameSpaceMatrix[look.X, look.Y].ContainsSnakeSegment ? 1 : 0;
        }

        //orientation
        int orientationIndex = 
            xDiff == 1 ? 3 :
            xDiff == -1 ? 4 :
            yDiff == 1 ? 5 : 6;
        vision[orientationIndex] = 1;

        //food
        int? horizontalFoodIndex = null;
        if (foodXDiff != 0)
            horizontalFoodIndex = xDiff == 1 ? 9 : 7;
        int? verticalFoodIndex = null;
        if (foodYDiff != 0)
            verticalFoodIndex = yDiff == 1 ? 10 : 8;
        if(horizontalFoodIndex != null)
            vision[horizontalFoodIndex.Value] = 1;
        if(verticalFoodIndex != null)
            vision[verticalFoodIndex.Value] = 1;

        for (int i = 0; i < 3; i++)
        {
            Coord look = new Coord
            {
                X = (int)Mathf.Round(Segments[0].Position.X + offsets[i][0]),
                Y = (int)Mathf.Round(Segments[0].Position.Y + offsets[i][1]),
            };

            //adjacent food check
            vision[i + 11] =
                look.X < 0 || look.X >= SnakeGame.XSize ||
                look.Y < 0 || look.Y >= SnakeGame.YSize ||
                SnakeGame.GameSpaceMatrix[look.X, look.Y].ContainsFood ? 1 : 0;
        }

        return vision;
    }

    bool CheckAteFood()
    {
        List<SnakeGameSquare> foodSquares = SnakeGame.GameSpaceMatrix
            .Cast<SnakeGameSquare>().ToList()
            .Where(square =>
                square.ContainsFood &&
                square.Position.X == Segments[0].Position.X &&
                square.Position.Y == Segments[0].Position.Y
            ).ToList();

        if (!foodSquares.Any()) return false;

        SnakeGameSquare foodSquare = foodSquares.First();
        SnakeGame.GameSpaceMatrix[foodSquare.Position.X, foodSquare.Position.Y].ContainsFood = false;
        SnakeGame.GameSpaceMatrix[foodSquare.Position.X, foodSquare.Position.Y].SetColor(Color.white);
        Segments.Add(new SnakeGameSquare(lastTailCoord.X, lastTailCoord.Y, SnakeGame.SquareObject, Color.red));

        return true;
    }

    public void RenderSnake()
    {
        for (int x = 0; x < SnakeGame.XSize; x++)
            for (int y = 0; y < SnakeGame.YSize; y++)
                SnakeGame.GameSpaceMatrix[x, y].ContainsSnakeSegment = false;
        for (int i = 0; i < Segments.Count; i++)
        {
            SnakeGame.GameSpaceMatrix[Segments[i].Position.X, Segments[i].Position.Y].ContainsSnakeSegment = true;
            Segments[i].GameSpaceObject.transform.position =
                SnakeGameSquare.GetGameSpaceLocation(Segments[i].Position.X, Segments[i].Position.Y);
        }
    }
}

public struct SnakeMoveResult
{
    public bool OutOfBounds { get; set; }
    public bool HitSelf { get; set; }
    public bool AteFood { get; set; }
}

public enum Move
{
    Left = 0,
    Straight = 1,
    Right = 2
}