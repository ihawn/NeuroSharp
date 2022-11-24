using System.Net;
using System.Runtime.CompilerServices;
using NeuroSharp.Training;
using System.Net.Http.Json;

namespace NeurosharpBlazorWASM.Models;

public class NetworkStorage
{
    public Network CharacterClassificationModel { get; set; }

    public NetworkStorage()
    {
        LoadCharacterModel();
    }

    public async void LoadCharacterModel()
    {
        CharacterClassificationModel = Network.DeserializeNetworkJSON(
            File.ReadAllText($"{System.IO.Directory.GetCurrentDirectory()}{@"NetworkModels/characters_model.json"}")
        );
        
    }
}