using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class SnakeGame : MonoBehaviour
{
    public int XSize;
    public int YSize;
    public GameObject SquareObject;
    public Snake Snake;
    int SnakeStartLength;
    public bool FoodExists
    {
        get
        {
            return GameSpaceMatrix.Cast<SnakeGameSquare>().Any(x => x.ContainsFood);
        }
    }

    public SnakeGameSquare[,] GameSpaceMatrix { get; private set; }


    public SnakeGame(int xSize, int ySize, int snakeStartLength, GameObject squareObject)
    {
        if (xSize < 6 || ySize < 6 || xSize % 2 == 1 || ySize % 2 == 1)
            throw new System.Exception("Snake game must be at least 6x6 and dimensions must be even");

        SquareObject = squareObject;
        SnakeStartLength = snakeStartLength;
        GameSpaceMatrix = new SnakeGameSquare[xSize, ySize];

        XSize = xSize;
        YSize = ySize;

        for(int x = 0; x < xSize; x++)
            for(int y = 0; y < ySize; y++)
                GameSpaceMatrix[x, y] = new SnakeGameSquare(x, y, SquareObject, Color.white);

        Snake = new Snake(snakeStartLength, this);
    }

    public void ResetSnake()
    {
        foreach(SnakeGameSquare segment in Snake.Segments)
            Destroy(segment.GameSpaceObject);
        Snake = new Snake(SnakeStartLength, this);
        Snake.RenderSnake();
    }

    public void SpawnFood()
    {
        List<Coord> foodPlacementCandidates = 
            GameSpaceMatrix.Cast<SnakeGameSquare>()
            .Where(x => !x.ContainsSnakeSegment).Select(x => x.Position).ToList();

        Coord nextFoodPosition = foodPlacementCandidates[Random.Range(0, foodPlacementCandidates.Count)];
        GameSpaceMatrix[nextFoodPosition.X, nextFoodPosition.Y].ContainsFood = true;
        GameSpaceMatrix[nextFoodPosition.X, nextFoodPosition.Y].SetColor(Color.green);
    }
}

public struct Coord
{
    public int X { get; set; }
    public int Y { get; set; }
}

public enum MoveResult
{
    Nothing = 0,
    Death = 1,
    Eat = 2
}
