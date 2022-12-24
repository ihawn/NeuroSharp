using Microsoft.AspNetCore.Mvc;
using MathNet.Numerics.LinearAlgebra;
using NeuroSharp.Data;
using NeuroSharp.Training;
using NeurosharpBlazorWASM.Models;

namespace NeurosharpBlazorWASM.Server.Controllers;

[ApiController]
[Route("[controller]")]
public class SentimentAnalysisController : ControllerBase
{
    [HttpGet]
    public List<double> Get(string fieldValue)
    {
        Network network = NetworkStorage.Networks["SentimentAnalysis"];
        Vector<double> encodedSentence = NLPDataUtils.GetEncodedWordStream(fieldValue, network.Data, 45);

        return network.Predict(encodedSentence).ToList();
    }
}
