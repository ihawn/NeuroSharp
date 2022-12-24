
using NeuroSharp.Training;

namespace NeurosharpBlazorWASM.Models;

public class NetworkStorage
{
    private static NetworkStorage _instance = null;
    private Dictionary<string, Network> _networks { get; set; }

    public static NetworkStorage Networks
    {
        get
        {
            if (_instance == null)
            {
                _instance = new NetworkStorage();
                _instance._networks = new Dictionary<string, Network>();
            }

            return _instance;
        }
    }

    public Network this[string index] => _networks[index];

    public void LoadAllNetworkModels()
    {
        _networks["CharacterRecognition"] = Network.DeserializeNetworkJSON(
            File.ReadAllText($"{System.IO.Directory.GetCurrentDirectory()}{@"/NetworkModels/characters_model.json"}")
        );
        _networks["SentimentAnalysis"] = Network.DeserializeNetworkJSON(
            File.ReadAllText($"{System.IO.Directory.GetCurrentDirectory()}{@"/NetworkModels/sentiment_analysis_model.json"}")
        );
    }
}