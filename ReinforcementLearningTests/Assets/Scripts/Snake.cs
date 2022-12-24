using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class Snake
{
    public SnakeGame SnakeGame { get; private set; }
    public List<SnakeGameSquare> Segments { get; set; }
    public int Length { get { return Segments.Count; } }

    public Snake(int startingLength, SnakeGame parentGame)
    {
        if (startingLength < 2)
            throw new System.Exception("Snake must start with at least 2 segments");

        SnakeGame = parentGame;

        Segments = new List<SnakeGameSquare>();
        for (int x = startingLength - 1; x >= 0; x--)
            Segments.Add(new SnakeGameSquare(x, parentGame.YSize / 2, parentGame.SquareObject, Color.red));
    }

    public bool MoveSnake(Move move) //returns out of bounds or not
    {
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
            return true;

        for (int i = Segments.Count - 1; i > 0; i--)
        {
            Segments[i].Position.X = Segments[i - 1].Position.X;
            Segments[i].Position.Y = Segments[i - 1].Position.Y;
        }

        Segments[0].Position = newHeadPos;

        RenderSnake();
        return false;
    }

    void RenderSnake()
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

public enum Move
{
    Left = 0,
    Straight = 1,
    Right = 2
}