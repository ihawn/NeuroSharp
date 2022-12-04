using System.Net;
using System.Runtime.CompilerServices;
using NeuroSharp.Training;
using System.Net.Http.Json;

namespace NeurosharpBlazorWASM.Models;

public class NetworkStorage
{
    private static NetworkStorage _instance = null;
    private Dictionary<string, Network> _settings { get; set; }

    public static NetworkStorage Networks
    {
        get
        {
            if (_instance == null)
                _instance = new NetworkStorage();
            return _instance;
        }
    }

    public Network this[string index] => _settings[index];

    public void LoadAllNetworkModels()
    {
        _settings["CharacterRecognition"] = Network.DeserializeNetworkJSON(
            File.ReadAllText($"{System.IO.Directory.GetCurrentDirectory()}{@"NetworkModels/characters_model_ab.json"}")
        );
    }
}