using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NeuroSharp;
using NeuroSharp.Enumerations;
using MathNet.Numerics.LinearAlgebra;

public class QLearner
{
    public NeuroSharp.Training.Network Network { get; private set; }
    public SnakeGame SnakeGame { get; private set; }

    public QLearner(SnakeGame snakeGame)
    {
        Network = new NeuroSharp.Training.Network(snakeGame.XSize * snakeGame.YSize);
        Network.Layers.Add(new FullyConnectedLayer(256));
        Network.Layers.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Layers.Add(new FullyConnectedLayer(128));
        Network.Layers.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Layers.Add(new FullyConnectedLayer(64));
        Network.Layers.Add(new ActivationLayer(ActivationType.Tanh));
        Network.Layers.Add(new FullyConnectedLayer(3));
        Network.Layers.Add(new SoftmaxActivationLayer());
        Network.UseLoss(LossType.CategoricalCrossentropy);
    }

}
