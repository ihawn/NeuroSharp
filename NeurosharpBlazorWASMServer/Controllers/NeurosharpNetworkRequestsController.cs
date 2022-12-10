using Microsoft.AspNetCore.Mvc;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Training;
using NeurosharpBlazorWASM.Models;

namespace NeurosharpBlazorWASM.Server.Controllers;

[ApiController]
[Route("[controller]")]
public class NeurosharpNetworkRequestsController : ControllerBase
{
    [HttpGet]
    public List<double> Get(string delimitedPixelValues)
    {
        Network network = NetworkStorage.Networks["CharacterRecognition"];
        
        Vector<double> prediction = network.Predict(
            Vector<double>.Build.DenseOfEnumerable(
                delimitedPixelValues.Split(',').Select(x => double.Parse(x))
            )
        );
        
       return prediction.ToList();
    }

    /*[HttpGet]
    public List<double> GetSentiment(string fieldValue)
    {
        Network network = NetworkStorage.Networks["SentimentAnalysis"];

        return new List<double>();
    }*/
}
