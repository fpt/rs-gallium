using GalliumWinUI.Models;
using GalliumWinUI.Services;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using System.Collections.ObjectModel;
using Windows.Graphics;
using Windows.System;

namespace GalliumWinUI;

public sealed partial class MainWindow : Window
{
    private readonly AgentService _agent = new();
    private readonly TtsService _tts = new();
    private readonly SttService _stt = new();
    private readonly ObservableCollection<ChatMessage> _messages = [];
    private bool _busy;

    public MainWindow()
    {
        InitializeComponent();
        MessageList.ItemsSource = _messages;
        AppWindow.Resize(new SizeInt32(900, 680));
        _ = InitAgentAsync();
    }

    private async Task InitAgentAsync()
    {
        StatusLabel.Text = "Loading…";
        try
        {
            var cfg = AppConfig.Load();
            await _agent.InitializeAsync(cfg);
            ModelLabel.Text = cfg.Llm.Model;
            StatusLabel.Text = "Ready";
            SendButton.IsEnabled = true;
        }
        catch (Exception ex)
        {
            StatusLabel.Text = $"Error: {ex.Message}";
        }
    }

    private async void SendButton_Click(object sender, RoutedEventArgs e)
        => await SendAsync(InputBox.Text);

    private async void InputBox_KeyDown(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter && !_busy)
            await SendAsync(InputBox.Text);
    }

    private async Task SendAsync(string text)
    {
        if (_busy || !_agent.IsReady || string.IsNullOrWhiteSpace(text)) return;
        _busy = true;
        SendButton.IsEnabled = false;
        InputBox.Text = "";

        _messages.Add(new ChatMessage(text, isUser: true));
        ScrollToBottom();

        try
        {
            var resp = await _agent.StepAsync(text);
            _messages.Add(new ChatMessage(resp.Content, isUser: false));
            StatusLabel.Text = resp.TotalTokens > 0
                ? $"in={resp.InputTokens} out={resp.OutputTokens} ctx={resp.ContextPercent:F0}%"
                : "Ready";
            if (TtsToggle.IsChecked == true)
                await _tts.SpeakAsync(resp.Content);
        }
        catch (Exception ex)
        {
            _messages.Add(new ChatMessage($"[Error] {ex.Message}", isUser: false));
        }
        finally
        {
            _busy = false;
            SendButton.IsEnabled = true;
            ScrollToBottom();
        }
    }

    private void ScrollToBottom()
    {
        HistoryScroller.UpdateLayout();
        HistoryScroller.ChangeView(null, HistoryScroller.ScrollableHeight, null);
    }

    private async void MicButton_Click(object sender, RoutedEventArgs e)
    {
        var text = await _stt.RecognizeOnceAsync();
        if (!string.IsNullOrWhiteSpace(text))
        {
            InputBox.Text = text;
            await SendAsync(text);
        }
    }
}
