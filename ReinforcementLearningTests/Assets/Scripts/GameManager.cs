using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Linq;

public class GameManager : MonoBehaviour
{
    public SnakeGame SnakeGame;
    public GameObject SquareObject;
    public int GameSizeX = 6;
    public int GameSizeY = 6;
    public int SnakeStartLength = 3;

    public float TimeStep = 1;
    public float MoveTimer = 0;
    public float GameTick = 1;

    Snake Snake { get { return SnakeGame.Snake; } }

    void Start()
    {
        SnakeGame = new SnakeGame(GameSizeX, GameSizeY, SnakeStartLength, SquareObject);
    }

    void Update()
    {
        MoveTimer += Time.deltaTime * TimeStep;
        if(MoveTimer > GameTick)
        {
            MoveTimer = 0;
            bool outOfBounds = Snake.MoveSnake((Move)Random.Range(0, 3));
            if (outOfBounds) SnakeGame.ResetSnake();
        }
    }
}
