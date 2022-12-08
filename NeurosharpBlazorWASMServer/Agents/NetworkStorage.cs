
using NeuroSharp.Training;

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
            {
                _instance = new NetworkStorage();
                _instance._settings = new Dictionary<string, Network>();
            }

            return _instance;
        }
    }

    public Network this[string index] => _settings[index];

    public void LoadAllNetworkModels()
    {
        _settings["CharacterRecognition"] = Network.DeserializeNetworkJSON(
            File.ReadAllText($"{System.IO.Directory.GetCurrentDirectory()}{@"/NetworkModels/characters_model.json"}")
        );
    }
}