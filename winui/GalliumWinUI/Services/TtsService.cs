using Windows.Media.Playback;
using Windows.Media.SpeechSynthesis;

namespace GalliumWinUI.Services;

internal sealed class TtsService : IDisposable
{
    private readonly SpeechSynthesizer _synth = new();
    private readonly MediaPlayer _player = new();
    private TaskCompletionSource? _tcs;

    public TtsService()
    {
        _player.MediaEnded += (_, _) => _tcs?.TrySetResult();
        _player.MediaFailed += (_, _) => _tcs?.TrySetResult();
    }

    public async Task SpeakAsync(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return;
        _player.Source = null;
        var stream = await _synth.SynthesizeTextToStreamAsync(text);
        _tcs = new TaskCompletionSource();
        _player.SetStreamSource(stream);
        _player.Play();
        await _tcs.Task;
    }

    public void Stop() => _player.Pause();

    public void Dispose()
    {
        _synth.Dispose();
        _player.Dispose();
    }
}
