using Windows.Media.SpeechRecognition;

namespace GalliumWinUI.Services;

internal sealed class SttService
{
    /// Push-to-talk: shows the system speech recognition UI, returns transcribed text.
    /// RecognizeWithUIAsync works in unpackaged apps; silent RecognizeAsync may require package identity.
    public async Task<string?> RecognizeOnceAsync()
    {
        var recognizer = new SpeechRecognizer();
        try
        {
            await recognizer.CompileConstraintsAsync();
            var result = await recognizer.RecognizeWithUIAsync();
            return result.Status == SpeechRecognitionResultStatus.Success ? result.Text : null;
        }
        catch
        {
            return null;
        }
        finally
        {
            recognizer.Dispose();
        }
    }
}
