using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MathNet.Numerics.LinearAlgebra;

public class GameState
{
    public GameState(GameManager gameManager, float offset, float scale) { Offset = offset; Scale = scale; gm = gameManager; }

    GameManager gm { get; set; }
    float Offset { get; set; }
    float Scale { get; set; }

    Direction LastDirectionTaken
    {
        get
        {
            return gm.NextDirection;
        }
    }

    public AbsoluteDirection AbsoluteDirection
    {
        get
        {
            if (gm.Snake.AbsoluteDirectionVector.x > 0) return AbsoluteDirection.Right;
            if (gm.Snake.AbsoluteDirectionVector.x < 0) return AbsoluteDirection.Left;
            if (gm.Snake.AbsoluteDirectionVector.y > 0) return AbsoluteDirection.Up;
            else return AbsoluteDirection.Down;
        }
    }

    Vector3 HeadPt { get { return gm.Snake.Segments[0].transform.position; } }
    public Vector3 LeftPoint { get { return new Vector3(Mathf.Round((HeadPt.x - Offset) / Scale) * Scale, HeadPt.y, 0); } }
    public Vector3 RightPoint { get { return new Vector3(Mathf.Round((HeadPt.x + Offset) / Scale) * Scale, HeadPt.y, 0); } }
    public Vector3 UpPoint { get { return new Vector3(HeadPt.x, Mathf.Round((HeadPt.y + Offset) / Scale) * Scale, 0); } }
    public Vector3 DownPoint { get { return new Vector3(HeadPt.x, Mathf.Round((HeadPt.y - Offset) / Scale) * Scale, 0); } }

    public bool DangerStraight
    {
        get
        {
            return (AbsoluteDirection == AbsoluteDirection.Right && gm.Snake.IsCollision(RightPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Left && gm.Snake.IsCollision(LeftPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Up && gm.Snake.IsCollision(UpPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Down && gm.Snake.IsCollision(DownPoint));
        }
    }

    public bool DangerRight
    {
        get
        {
            return (AbsoluteDirection == AbsoluteDirection.Up && gm.Snake.IsCollision(RightPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Down && gm.Snake.IsCollision(LeftPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Left && gm.Snake.IsCollision(UpPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Right && gm.Snake.IsCollision(DownPoint));
        }
    }
    public bool DangerLeft
    {
        get
        {
            return (AbsoluteDirection == AbsoluteDirection.Down && gm.Snake.IsCollision(RightPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Up && gm.Snake.IsCollision(LeftPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Right && gm.Snake.IsCollision(UpPoint)) ||
                   (AbsoluteDirection == AbsoluteDirection.Left && gm.Snake.IsCollision(DownPoint));
        }
    }
    public bool[] FoodLocation
    {
        get
        {
            //food is left, food is right, etc
            return new bool[]
            {
                gm.Food.transform.position.x < gm.Snake.Segments[0].transform.position.x,
                gm.Food.transform.position.x > gm.Snake.Segments[0].transform.position.x,
                gm.Food.transform.position.y < gm.Snake.Segments[0].transform.position.y,
                gm.Food.transform.position.y > gm.Snake.Segments[0].transform.position.y
            };
        }
    }

    //Vector for the neural net
    public Vector<float> StateVector
    {
        get
        {
            float[] f = new float[]
            {
                DangerStraight ? 1 : 0,
                DangerRight ? 1 : 0,
                DangerLeft ? 1 : 0,
                AbsoluteDirection == AbsoluteDirection.Up ? 1 : 0,
                AbsoluteDirection == AbsoluteDirection.Down ? 1 : 0,
                AbsoluteDirection == AbsoluteDirection.Left ? 1 : 0,
                AbsoluteDirection == AbsoluteDirection.Right ? 1 : 0,
                FoodLocation[0] ? 1 : 0,
                FoodLocation[1] ? 1 : 0,
                FoodLocation[2] ? 1 : 0,
                FoodLocation[3] ? 1 : 0
            };

            return Vector<float>.Build.DenseOfArray(f);
        }
    }
}
