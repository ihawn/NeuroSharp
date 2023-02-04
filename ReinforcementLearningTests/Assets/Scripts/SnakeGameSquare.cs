using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SnakeGameSquare : MonoBehaviour
{
    public Vector2 GamespaceLocation;
    public Coord Position;
    public bool ContainsSnakeSegment;
    public bool ContainsFood;
    public GameObject GameSpaceObject;

    public SnakeGameSquare(int x, int y, GameObject gameObject, Color color, float worldScaleFactor = 1)
    {
        Position = new Coord { X = x, Y = y };
        GamespaceLocation = GetGameSpaceLocation(x, y);
        GameSpaceObject = Instantiate(gameObject, GamespaceLocation, Quaternion.identity);
        GameSpaceObject.GetComponent<SpriteRenderer>().color = color;
    }

    public void SetColor(Color color)
    {
        GameSpaceObject.GetComponent<SpriteRenderer>().color = color;
    }    

    public static Vector2 GetGameSpaceLocation(int x, int y, float worldScaleFactor = 1)
    {
        return new Vector2(x * worldScaleFactor, y * worldScaleFactor);
    }
}
